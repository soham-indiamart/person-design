"""Voice demo server: browser mic -> Saaras -> live profile + persona -> Sarvam LLM -> Bulbul.

    python -m vani_persona.ps04.server [--port 8765] [--root .]
    then open http://localhost:8765

Standard library only. Binds to localhost by default. Seller data stays on this machine;
only the call audio/text of the demo itself is sent to Sarvam.

Hosted (e.g. Render): ``HOST``/``PORT`` set the bind address, and ``DEMO_PASSWORD`` (if set)
puts every page and API behind HTTP basic auth so strangers cannot spend Sarvam credits.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vani_persona import REPO_ROOT
from vani_persona.ps04 import sarvam
from vani_persona.ps04.evidence import load_evidence
from vani_persona.ps04.flows import FLOWS, buyer_spec
from vani_persona.ps04.generator import generate
from vani_persona.ps04.live import CallSession
from vani_persona.ps04.live_personas import PERSONAS, starting_persona
from vani_persona.ps04.seller_file import FORMATS, SAMPLE, build_spec, detect_format, parse
from vani_persona.ps04.use_cases import USE_CASES, apply_use_case, resolve

STATIC = Path(__file__).parent / "static"
ART = REPO_ROOT / "artifacts" / "ps04"
DEMO_ORDER = ["new_receptive", "time_pressed", "english_first", "established_formal", "guarded", "already_engaged"]


class State:
    def __init__(self, root: Path):
        self.root = root
        self.sessions: dict[str, CallSession] = {}
        demo_p = ART / "demo_sellers.json"   # generated from the PS07 export; absent on a fresh deploy
        self.demo = json.loads(demo_p.read_text(encoding="utf-8")) if demo_p.exists() else {}
        shared_p = ART / "shared_policy.json"
        self.shared = json.loads(shared_p.read_text(encoding="utf-8")) if shared_p.exists() else None
        self._sellers = None
        self._lock = threading.Lock()
        self.file_name, self.file_rows, self.file_format = "", {}, None
        if SAMPLE.exists():
            self.load_file(SAMPLE.read_bytes(), SAMPLE.name)

    def load_file(self, data: bytes, name: str) -> None:
        rows = parse(data, name)
        if not rows:
            raise ValueError("no seller rows found in the file")
        self.file_name, self.file_rows = name, {"f:" + r["_key"]: r for r in rows}
        self.file_format = FORMATS[detect_format(data, name)]

    def file_summary(self) -> dict:
        """Cards plus how the file was read and which use case each seller was routed to."""
        cards = self.file_cards()
        routed: dict[str, int] = {}
        for c in cards:
            if c.get("use_case"):
                routed[c["use_case"]["name"]] = routed.get(c["use_case"]["name"], 0) + 1
        first = next((c["key"] for c in cards if c.get("use_case") and not c.get("dnc")), None)
        return {"name": self.file_name, "format": self.file_format, "sellers": cards,
                "routing": routed, "first_callable": first}

    def file_cards(self) -> list[dict]:
        ev = load_evidence()
        return [_card(k, build_spec(r, ev)) for k, r in self.file_rows.items()]

    def sellers(self):
        with self._lock:
            if self._sellers is None:
                from vani_persona.ps04.data import load_sellers
                df = load_sellers(self.root)
                self._sellers = df.set_index("fk_glusr_usr_id", drop=False)
            return self._sellers

    def spec_for(self, key: str, use_case: str | None = None) -> dict:
        """Spec for a seller; ``use_case`` (if given) overrides the one in / inferred from the seller file."""
        if key in self.file_rows:
            return build_spec(self.file_rows[key], load_evidence(), use_case)
        spec = self._base_spec(key)
        return apply_use_case(spec, {}, use_case) if resolve(use_case) else spec

    def _base_spec(self, key: str) -> dict:
        if key == "unknown":
            spec = generate({"fk_glusr_usr_id": None}, load_evidence())
            spec["seller_type"] = "unknown"
            spec["summary"] = "Unknown seller: no seller-file data; Hinglish defaults (low confidence)"
            return spec
        if key in self.demo:
            return self.demo[key]
        df = self.sellers()
        sid = int(key)
        if sid not in df.index:
            raise KeyError(f"seller {sid} not in the seller file")
        row = {k: (None if isinstance(v, float) and v != v else v) for k, v in df.loc[sid].to_dict().items()}
        return generate(row, load_evidence())


def _card(key: str, spec: dict) -> dict:
    if "voice" not in spec:
        return {"key": key, "seller_type": spec["seller_type"], "summary": "Do-not-call: no persona generated",
                "dnc": True}
    start, why = starting_persona(spec)
    uc = spec.get("use_case")
    return {
        "key": key, "seller_type": spec["seller_type"], "summary": spec["summary"],
        "use_case": ({"id": uc["id"], "name": uc["name"], "reason": uc["reason"], "archetype": uc["archetype"]["label"],
                      "requirement": " ".join(uc["context"]), "who": " / ".join(
                          x for x in (uc.get("company"), uc.get("seller_name")) if x)} if uc else None),
        "confidence": spec["overall_confidence"], "start_persona": PERSONAS[start].name, "start_reason": why,
        "settings": {
            "speaker": spec["voice"]["speaker"]["value"], "pace": spec["voice"]["pace"]["value"],
            "language": spec["language"]["primary"]["value"], "mix": spec["language"]["hinglish_mix"]["value"],
            "formality": spec["language"]["formality"]["value"], "warmth": spec["tone"]["warmth"]["value"],
            "empathy": spec["tone"]["empathy"]["value"],
        },
    }


def make_handler(state: State, password: str | None = None):
    expected = "Basic " + base64.b64encode(f"vani:{password}".encode()).decode() if password else None

    class H(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if expected is None or hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="VANI persona demo"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def log_message(self, fmt, *args):  # keep the console readable
            if "/api/" in (str(args[0]) if args else ""):
                super().log_message(fmt, *args)

        def _set_cors(self):
            # The UI is served from this server, so it needs no CORS; a wildcard would let any
            # website drive the API (and spend credits). Allow one extra origin only if configured.
            origin = os.environ.get("CORS_ORIGIN")
            if not origin:
                return
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self._set_cors()
            self.end_headers()

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._set_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _out(self, r: dict) -> dict:
            r = dict(r)
            c = sarvam.BUDGET.status()
            r["credits"] = {"spent_estimate": c["spent_estimate"], "limit": c["limit"], "remaining": c["remaining"]}
            if r.get("audio"):
                r["audio"] = base64.b64encode(r["audio"]).decode()
            return r

        def _stream_start(self):
            """Open an NDJSON response; returns ``send(obj)``."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._set_cors()
            self.end_headers()
            self.close_connection = True

            def send(obj):
                self.wfile.write(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
                self.wfile.flush()
            return send

        def _stream(self, s: CallSession, r: dict, extra: dict | None = None, send=None):
            """NDJSON: the turn (text, analysis, switches, profile) at once, then each sentence's audio
            as soon as it is synthesised, so the browser starts playing the first sentence early."""
            send = send or self._stream_start()
            try:
                send({"type": "turn", **(extra or {}), **self._out(r)})
                if r.get("skipped") or not r.get("reply"):
                    return send({"type": "done"})
                timings = {}
                try:
                    for i, wav, ms in s.speech_chunks(r["reply"]):
                        if i == 0:
                            timings["first_audio_ms"] = ms + r["timings"].get("reply_total_ms", 0)
                        send({"type": "audio", "i": i, "audio": base64.b64encode(wav).decode()})
                        timings["tts_ms"] = ms
                except sarvam.SarvamError as e:
                    send({"type": "error", "error": f"Sarvam: {e}"})
                send({"type": "done", "timings": timings, "credits": self._out({})["credits"]})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser hung up mid-reply

        def do_GET(self):
            if self.path == "/healthz":
                return self._json({"ok": True})
            if not self._authorized():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    body = (STATIC / "index.html").read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self._set_cors()
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/sellers":
                    keys = [k for k in DEMO_ORDER if k in state.demo]
                    cards = [_card("unknown", state.spec_for("unknown"))] + [_card(k, state.demo[k]) for k in keys]
                    personas = [{"id": p.id, "name": p.name, "state": p.seller_state, "description": p.description,
                                 "colour": p.colour} for p in PERSONAS.values()]
                    modes = [{"mode": f.mode, "label": f.label, "stages": [st.name for st in f.stages]}
                             for f in FLOWS.values()]
                    use_cases = [{"id": u.id, "name": u.name, "goal": u.goal, "metric": u.metric}
                                 for u in USE_CASES.values()]
                    self._json({"sellers": cards, "personas": personas, "modes": modes, "use_cases": use_cases,
                                "file": state.file_summary(),
                                "credits": sarvam.BUDGET.status(),
                                "languages": {"speak_and_understand": sarvam.TTS_LANGUAGES,
                                              "understand_only": sarvam.STT_ONLY_LANGUAGES}})
                elif u.path == "/api/spec":
                    try:
                        self._json(state.spec_for(q["seller"][0], q.get("use_case", [None])[0]))
                    except (KeyError, ValueError) as e:
                        self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
                elif u.path == "/api/log":
                    s = state.sessions.get(q.get("session", [""])[0])
                    if not s:
                        return self._json({"error": "no such session"}, HTTPStatus.NOT_FOUND)
                    self._json({"call_id": s.id, "seller_type": s.spec["seller_type"],
                                "use_case": s.spec.get("use_case"), "flow": s.flow_status(),
                                "start_settings": s.base.as_dict(), "final_settings": s.settings.as_dict(),
                                "persona_timeline": s.timeline, "start_reason": s.start_reason,
                                "live_profile": s.profile.as_dict(), "profile_history": s.profile.history,
                                "analyst_verdicts": [v.as_dict() for v in s.verdicts],
                                "switch_log": s.switch_log, "outcome": s.outcome,
                                "transcript": [{"role": t.role, "text": t.text} for t in s.turns]})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            if not self._authorized():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/upload":
                    if q.get("sample", ["0"])[0] == "1":
                        state.load_file(SAMPLE.read_bytes(), SAMPLE.name)
                    else:
                        try:
                            state.load_file(self._body(), q.get("name", ["sellers.csv"])[0])
                        except Exception as e:
                            return self._json({"error": f"Could not read the seller file: {e}"},
                                              HTTPStatus.BAD_REQUEST)
                    self._json(state.file_summary())
                elif u.path == "/api/start":
                    req = json.loads(self._body() or b"{}")
                    mode = req.get("mode") or "auto"
                    use_case = mode if resolve(mode) and mode != "seller" else None
                    spec = state.spec_for(str(req.get("seller") or "unknown"), use_case)
                    if "voice" not in spec:
                        return self._json({"error": "This seller asked not to be called; the call is suppressed."},
                                          HTTPStatus.CONFLICT)
                    if mode == "buyer":
                        spec = buyer_spec(spec, req.get("product"))
                    s = CallSession(spec, start_persona=req.get("persona") or None)
                    state.sessions[s.id] = s
                    stream = bool(req.get("stream"))
                    r = s.open(stream=stream)
                    s.save()
                    if stream:
                        return self._stream(s, r, {"session": s.id, "spec": spec})
                    self._json({"session": s.id, "spec": spec, **self._out(r)})
                elif u.path == "/api/turn":
                    s = state.sessions[q["session"][0]]
                    stream = q.get("stream", ["0"])[0] == "1"
                    flags = {k for k in ("interrupted", "nudge", "flush") if q.get(k, ["0"])[0] == "1"}
                    ctype = self.headers.get("Content-Type", "")
                    body = self._body()
                    send, on_ack = None, None
                    if stream:
                        # Start the response now: an instant "achha" plays while the reply is written.
                        send = self._stream_start()

                        def on_ack(text, wav):
                            try:
                                send({"type": "audio", "i": -1, "ack": text, "audio": base64.b64encode(wav).decode()})
                            except OSError:
                                pass   # the browser hung up; the turn still completes for the log
                    try:
                        with s.lock:
                            if "interrupted" in flags:
                                s.mark_interrupted()
                            if "nudge" in flags:
                                r = s.nudge(stream=stream)
                            elif "flush" in flags:
                                r = s.flush(stream=stream, on_ack=on_ack)
                            elif ctype.startswith("application/json"):
                                req = json.loads(body)
                                r = s.seller_text(req["text"], req.get("language"), stream=stream,
                                                  hold_if_unfinished=bool(req.get("hold", False)), on_ack=on_ack)
                            else:
                                r = s.seller_audio(body, ctype or "audio/wav", stream=stream, on_ack=on_ack)
                    except Exception as e:
                        if not send:
                            raise
                        import traceback
                        traceback.print_exc()
                        msg = f"Sarvam: {e}" if isinstance(e, sarvam.SarvamError) else str(e)
                        try:
                            send({"type": "error", "error": msg})
                            send({"type": "done"})
                        except OSError:
                            pass
                        return
                    if stream:
                        return self._stream(s, r, send=send)
                    self._json(self._out(r))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except sarvam.SarvamError as e:
                self._json({"error": f"Sarvam: {e}"}, HTTPStatus.BAD_GATEWAY)
            except KeyError as e:
                self._json({"error": f"missing {e}"}, HTTPStatus.BAD_REQUEST)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    sarvam.api_key()  # fail fast if the key is missing
    password = os.environ.get("DEMO_PASSWORD") or None
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(State(args.root), password))
    print(f"VANI persona demo on http://{args.host}:{args.port}" + (" (password protected)" if password else ""))
    srv.serve_forever()


if __name__ == "__main__":
    main()
