"""Voice demo server: browser mic -> Saaras -> signals -> persona -> Sarvam LLM -> Bulbul.

    python -m vani_persona.ps04.server [--port 8765] [--root .]
    then open http://localhost:8765

Standard library only. Binds to localhost. Seller data stays on this machine; only the
call audio/text of the demo itself is sent to Sarvam.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vani_persona import REPO_ROOT
from vani_persona.ps04 import sarvam
from vani_persona.ps04.evidence import load_evidence
from vani_persona.ps04.generator import generate
from vani_persona.ps04.live import CallSession
from vani_persona.ps04.live_personas import PERSONAS, starting_persona

STATIC = Path(__file__).parent / "static"
ART = REPO_ROOT / "artifacts" / "ps04"
DEMO_ORDER = ["new_receptive", "time_pressed", "english_first", "established_formal", "guarded", "already_engaged"]


class State:
    def __init__(self, root: Path):
        self.root = root
        self.sessions: dict[str, CallSession] = {}
        self.demo = json.loads((ART / "demo_sellers.json").read_text())
        shared_p = ART / "shared_policy.json"
        self.shared = json.loads(shared_p.read_text()) if shared_p.exists() else None
        self._sellers = None
        self._lock = threading.Lock()

    def sellers(self):
        with self._lock:
            if self._sellers is None:
                from vani_persona.ps04.data import load_sellers
                df = load_sellers(self.root)
                self._sellers = df.set_index("fk_glusr_usr_id", drop=False)
            return self._sellers

    def spec_for(self, key: str) -> dict:
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
        return {"key": key, "seller_type": spec["seller_type"], "summary": "Do-not-call: no persona generated"}
    start, why = starting_persona(spec)
    return {
        "key": key, "seller_type": spec["seller_type"], "summary": spec["summary"],
        "confidence": spec["overall_confidence"], "start_persona": PERSONAS[start].name, "start_reason": why,
        "settings": {
            "speaker": spec["voice"]["speaker"]["value"], "pace": spec["voice"]["pace"]["value"],
            "language": spec["language"]["primary"]["value"], "mix": spec["language"]["hinglish_mix"]["value"],
            "formality": spec["language"]["formality"]["value"], "warmth": spec["tone"]["warmth"]["value"],
            "empathy": spec["tone"]["empathy"]["value"],
        },
    }


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console readable
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
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

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/sellers":
                keys = [k for k in DEMO_ORDER if k in state.demo]
                cards = [_card("unknown", state.spec_for("unknown"))] + [_card(k, state.demo[k]) for k in keys]
                personas = [{"id": p.id, "name": p.name, "state": p.seller_state, "description": p.description,
                             "colour": p.colour} for p in PERSONAS.values()]
                self._json({"sellers": cards, "personas": personas, "credits": sarvam.BUDGET.status(),
                            "languages": {"speak_and_understand": sarvam.TTS_LANGUAGES,
                                          "understand_only": sarvam.STT_ONLY_LANGUAGES}})
            elif u.path == "/api/spec":
                try:
                    self._json(state.spec_for(q["seller"][0]))
                except (KeyError, ValueError) as e:
                    self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
            elif u.path == "/api/log":
                s = state.sessions.get(q.get("session", [""])[0])
                if not s:
                    return self._json({"error": "no such session"}, HTTPStatus.NOT_FOUND)
                self._json({"call_id": s.id, "seller_type": s.spec["seller_type"],
                            "start_settings": s.base.as_dict(), "final_settings": s.settings.as_dict(),
                            "persona_timeline": s.timeline, "start_reason": s.start_reason,
                            "analyst_verdicts": [v.as_dict() for v in s.verdicts],
                            "switch_log": s.switch_log, "outcome": s.outcome,
                            "transcript": [{"role": t.role, "text": t.text} for t in s.turns]})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/start":
                    req = json.loads(self._body() or b"{}")
                    spec = state.spec_for(str(req["seller"]))
                    if "voice" not in spec:
                        return self._json({"error": "This seller asked not to be called; the call is suppressed."},
                                          HTTPStatus.CONFLICT)
                    s = CallSession(spec)
                    state.sessions[s.id] = s
                    r = s.open()
                    s.save()
                    self._json({"session": s.id, "spec": spec, **self._out(r)})
                elif u.path == "/api/turn":
                    s = state.sessions[q["session"][0]]
                    ctype = self.headers.get("Content-Type", "")
                    if ctype.startswith("application/json"):
                        req = json.loads(self._body())
                        r = s.seller_text(req["text"], req.get("language"))
                    else:
                        r = s.seller_audio(self._body(), ctype or "audio/wav")
                    self._json(self._out(r))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except sarvam.SarvamError as e:
                self._json({"error": f"Sarvam: {e}"}, HTTPStatus.BAD_GATEWAY)
            except KeyError as e:
                self._json({"error": f"missing {e}"}, HTTPStatus.BAD_REQUEST)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    sarvam.api_key()  # fail fast if the key is missing
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(State(args.root)))
    print(f"VANI persona demo on http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
