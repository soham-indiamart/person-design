"""Minimal Sarvam AI client (stdlib only): Saaras speech-to-text, Bulbul text-to-speech, chat.

The key is read from ``SARVAM_API_KEY`` (environment, or the gitignored ``.env`` in the
repo root). It is never logged.

Credit cap: every call is priced *before* it is sent, using Sarvam's published rates
(1 credit = Rs 1), and refused if the running total would pass ``SARVAM_CREDIT_LIMIT``
(default 400). The total is kept in ``artifacts/ps04/sarvam_usage.json``. It is an
estimate: sync it with the dashboard balance using

    python -m vani_persona.ps04.sarvam --usage
    python -m vani_persona.ps04.sarvam --set-spent 52.0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

from vani_persona import REPO_ROOT

BASE = "https://api.sarvam.ai"
STT_MODEL = "saaras:v3"
TTS_MODEL = "bulbul:v3"
CHAT_MODEL = "sarvam-105b-conversations"

# Saaras recognises these (plus "unknown" = auto-detect); Bulbul can speak the first 11.
TTS_LANGUAGES = {
    "hi-IN": "Hindi", "en-IN": "English", "bn-IN": "Bengali", "gu-IN": "Gujarati", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia", "pa-IN": "Punjabi", "ta-IN": "Tamil",
    "te-IN": "Telugu",
}
STT_ONLY_LANGUAGES = {
    "as-IN": "Assamese", "ur-IN": "Urdu", "ne-IN": "Nepali", "kok-IN": "Konkani", "ks-IN": "Kashmiri",
    "sd-IN": "Sindhi", "sa-IN": "Sanskrit", "sat-IN": "Santali", "mni-IN": "Manipuri", "brx-IN": "Bodo",
    "mai-IN": "Maithili", "doi-IN": "Dogri",
}
ALL_LANGUAGES = {**TTS_LANGUAGES, **STT_ONLY_LANGUAGES}


# Published prices (sarvam.ai/api-pricing, checked 2026-10-09), in credits (1 credit = Rs 1).
PRICE_STT_PER_HOUR = 30.0
PRICE_TTS_PER_1K_CHARS = 3.0
PRICE_LLM_IN_PER_M = 29.28
PRICE_LLM_OUT_PER_M = 73.20
USAGE_PATH = REPO_ROOT / "artifacts" / "ps04" / "sarvam_usage.json"


class SarvamError(RuntimeError):
    pass


class CreditLimitReached(SarvamError):
    pass


class Budget:
    """Running credit estimate with a hard cap, shared by every caller in this process."""

    def __init__(self, path=USAGE_PATH, limit: float | None = None):
        self.path = path
        self.limit = float(limit if limit is not None else os.environ.get("SARVAM_CREDIT_LIMIT", 400))
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"spent_estimate": 0.0, "by_api": {}, "calls": 0}

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def status(self) -> dict:
        d = self._load()
        return {"spent_estimate": round(d["spent_estimate"], 2), "limit": self.limit,
                "remaining": round(self.limit - d["spent_estimate"], 2), "calls": d["calls"], "by_api": d["by_api"]}

    def check(self, api: str, cost: float) -> None:
        with self._lock:
            spent = self._load()["spent_estimate"]
            if spent + cost > self.limit:
                raise CreditLimitReached(
                    f"Sarvam credit cap reached: ~{spent:.1f} of {self.limit:.0f} credits used; this {api} call "
                    f"(~{cost:.2f}) was not sent. Raise SARVAM_CREDIT_LIMIT to continue.")

    def record(self, api: str, cost: float) -> None:
        with self._lock:
            d = self._load()
            d["spent_estimate"] += cost
            d["calls"] += 1
            d["by_api"][api] = round(d["by_api"].get(api, 0.0) + cost, 4)
            self._save(d)

    def set_spent(self, value: float) -> None:
        with self._lock:
            d = self._load()
            d["spent_estimate"] = float(value)
            self._save(d)


BUDGET = Budget()


def audio_seconds(audio: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio)) as r:
            return r.getnframes() / float(r.getframerate())
    except (wave.Error, EOFError):
        return len(audio) / 16000.0  # compressed audio: assume ~128 kbps


def stt_cost(audio: bytes) -> float:
    return audio_seconds(audio) / 3600 * PRICE_STT_PER_HOUR


def tts_cost(text: str) -> float:
    return len(text) / 1000 * PRICE_TTS_PER_1K_CHARS


def llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1e6 * PRICE_LLM_IN_PER_M + completion_tokens / 1e6 * PRICE_LLM_OUT_PER_M


def api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("SARVAM_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SarvamError("SARVAM_API_KEY is not set (environment or .env)")
    return key


def _post(path: str, body: bytes, content_type: str, timeout: float = 30.0, retries: int = 2) -> dict:
    key = api_key()
    headers = {"api-subscription-key": key, "Authorization": f"Bearer {key}", "Content-Type": content_type}
    for attempt in range(retries + 1):
        req = urllib.request.Request(BASE + path, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode("utf-8", "replace")
            if e.code in (429, 500, 503) and attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise SarvamError(f"{path} -> HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise SarvamError(f"{path} -> {e}") from None
    raise SarvamError(f"{path} failed")


@dataclass
class Transcript:
    text: str
    language_code: str | None
    language_probability: float | None
    latency_ms: int


def _multipart(fields: dict[str, str], audio: bytes, filename: str, mime: str) -> tuple[bytes, str]:
    boundary = "----vani" + uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode() + audio + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(audio: bytes, mode: str = "transcribe", language_code: str = "unknown",
               filename: str = "turn.wav", mime: str = "audio/wav") -> Transcript:
    """Saaras speech-to-text. ``mode``: transcribe (native script) or translit (romanized)."""
    cost = stt_cost(audio)
    BUDGET.check("speech_to_text", cost)
    t0 = time.perf_counter()
    body, ctype = _multipart({"model": STT_MODEL, "mode": mode, "language_code": language_code},
                             audio, filename, mime)
    o = _post("/speech-to-text", body, ctype)
    BUDGET.record("speech_to_text", cost)
    return Transcript(o.get("transcript") or "", o.get("language_code"), o.get("language_probability"),
                      int((time.perf_counter() - t0) * 1000))


def synthesize(text: str, language_code: str, speaker: str, pace: float = 1.0,
               temperature: float = 0.6, sample_rate: int = 22050) -> tuple[bytes, int]:
    """Bulbul text-to-speech. Returns (wav bytes, latency ms)."""
    text = text[:2500]
    cost = tts_cost(text)
    BUDGET.check("text_to_speech", cost)
    t0 = time.perf_counter()
    body = {"text": text, "language_code": language_code, "model": TTS_MODEL, "speaker": speaker,
            "pace": round(min(max(pace, 0.5), 2.0), 2), "temperature": round(min(max(temperature, 0.01), 1.0), 2),
            "speech_sample_rate": sample_rate}
    o = _post("/text-to-speech", json.dumps(body).encode(), "application/json")
    BUDGET.record("text_to_speech", cost)
    return base64.b64decode(o["audios"][0]), int((time.perf_counter() - t0) * 1000)


def chat(messages: list[dict], max_tokens: int = 300, temperature: float = 0.3,
         json_mode: bool = False, timeout: float = 30.0, retries: int = 2) -> tuple[str, int]:
    """Sarvam chat completion. Returns (content, latency ms)."""
    est_in = sum(len(m.get("content") or "") for m in messages) // 2  # Indic scripts: ~2 chars/token
    BUDGET.check("llm", llm_cost(est_in, max_tokens))
    t0 = time.perf_counter()
    body: dict = {"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    o = _post("/v1/chat/completions", json.dumps(body).encode(), "application/json", timeout, retries)
    u = o.get("usage") or {}
    BUDGET.record("llm", llm_cost(u.get("prompt_tokens", est_in), u.get("completion_tokens", max_tokens)))
    content = (o["choices"][0]["message"].get("content") or "").strip()
    return content, int((time.perf_counter() - t0) * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sarvam credit usage (estimate)")
    ap.add_argument("--usage", action="store_true", help="print the running estimate")
    ap.add_argument("--set-spent", type=float, help="set the running total, e.g. from the dashboard")
    args = ap.parse_args()
    if args.set_spent is not None:
        BUDGET.set_spent(args.set_spent)
    print(json.dumps(BUDGET.status(), indent=2))


if __name__ == "__main__":
    main()
