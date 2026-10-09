"""Live call engine: seller audio -> Behaviour Analyst -> Persona Director -> reply -> Sarvam voice.

The seller spec (generator) sets the baseline: speaker, language, formality, plan, and
the starting persona. After every seller turn the Behaviour Analyst agent reads the
transcript and reports the seller's state; the Persona Director decides whether to switch
to another named live persona (``live_personas.py``). Language mirroring runs alongside,
independent of persona. Every switch is written to the switch log: the signal, the
detector and the seller's words, the change made, and why. The speaker, facts,
guardrails and call purpose never change.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import io
import json
import re
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

from vani_persona import REPO_ROOT
from vani_persona.features.transcript import normalize
from vani_persona.ps04 import sarvam
from vani_persona.ps04.analyst import Verdict, analyse
from vani_persona.ps04.director import PersonaDirector
from vani_persona.ps04.live_personas import PERSONAS, starting_persona
from vani_persona.ps04.signals import END_PATTERN, Detection, LanguageTracker

LOG_DIR = REPO_ROOT / "artifacts" / "ps04" / "switch_logs"
POOL = cf.ThreadPoolExecutor(max_workers=8)

MIX_TEXT = {
    "hindi_heavy": "Hindi-led Hinglish: mostly Hindi, common English business words kept in English.",
    "balanced": "Balanced Hinglish: Hindi sentence structure with plenty of English business words.",
    "english_heavy": "English, with a few common Hindi words like 'ji' only if natural.",
    "native": "The seller's language, with English business words where they would naturally use them.",
}
SCRIPT_TEXT = {
    "hi-IN": "Write Hindi words in Devanagari and English words in Latin script.",
    "en-IN": "Write in English (Latin script).",
}
LENGTH_WORDS = {"normal": 35, "short": 22, "very_short": 14}


@dataclass
class LiveSettings:
    language: str
    mix: str
    formality: str
    speaker: str
    pace: float
    expressiveness: float
    warmth: str
    empathy: str
    sentence_length: str = "normal"
    vocabulary: str = "everyday"
    focus: str = "follow the plan"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Turn:
    role: str
    text: str
    meta: dict = field(default_factory=dict)


def settings_from_spec(spec: dict) -> LiveSettings:
    v, lang, tone = spec["voice"], spec["language"], spec["tone"]
    return LiveSettings(
        language=lang["primary"]["value"], mix=lang["hinglish_mix"]["value"], formality=lang["formality"]["value"],
        speaker=v["speaker"]["value"], pace=float(v["pace"]["value"]), expressiveness=float(v["expressiveness"]["value"]),
        warmth=tone["warmth"]["value"], empathy=tone["empathy"]["value"],
        sentence_length="short" if spec["seller_type"] == "time_pressed" else "normal",
    )


DELIVERY_KEYS = ("pace", "expressiveness", "warmth", "empathy", "sentence_length", "vocabulary")


def apply_persona(s: LiveSettings, base: LiveSettings, persona_id: str) -> dict:
    """Set the delivery settings of ``persona_id`` (seller baseline + persona template).

    Language, mix, formality and speaker are left alone. Returns {setting: [old, new]}.
    """
    before = s.as_dict()
    template = PERSONAS[persona_id].delivery
    for k in DELIVERY_KEYS:
        setattr(s, k, template.get(k, getattr(base, k)))
    s.focus = PERSONAS[persona_id].behaviour
    after = s.as_dict()
    return {k: [before[k], after[k]] for k in DELIVERY_KEYS if before[k] != after[k]}


def apply_language(s: LiveSettings, base: LiveSettings, d: Detection) -> tuple[dict, str]:
    """Mirror the seller's language. Returns ({setting: [old, new]}, reason)."""
    before = s.as_dict()
    new = d.detail["to"]
    if new in sarvam.TTS_LANGUAGES:
        s.language = new
        s.mix = "english_heavy" if new == "en-IN" else base.mix if new == "hi-IN" else "native"
        reason = f"Seller switched to {sarvam.ALL_LANGUAGES.get(new, new)}: reply in it."
    else:
        reason = (f"Seller spoke {sarvam.ALL_LANGUAGES.get(new, new)}; Saaras understands it but Sarvam has no voice "
                  "for it, so reply in simple Hindi and offer an executive who speaks it.")
        s.mix, s.vocabulary = "hindi_heavy", "simple"
    after = s.as_dict()
    return {k: [before[k], after[k]] for k in after if before[k] != after[k]}, reason


class CallSession:
    def __init__(self, spec: dict, use_llm_signals: bool = True):
        self.id = uuid.uuid4().hex[:10]
        self.spec = spec
        self.base = settings_from_spec(spec)
        self.settings = copy.deepcopy(self.base)
        self.persona_id, self.start_reason = starting_persona(spec)
        apply_persona(self.settings, self.base, self.persona_id)
        self.director = PersonaDirector(self.persona_id)
        self.lang = LanguageTracker(self.settings.language)
        self.turns: list[Turn] = []
        self.switch_log: list[dict] = []
        self.verdicts: list[Verdict] = []
        self.timeline: list[dict] = [{"seller_turn": 0, "persona": self.persona_id}]
        self.escalate = False
        self.use_llm_signals = use_llm_signals
        self.t0 = time.time()
        self.ended = False
        self.outcome: dict = {}

    # ---- prompt -----------------------------------------------------------------------
    def system_prompt(self) -> str:
        sp, s = self.spec, self.settings
        plan = sp["conversation_plan"]
        lang_name = sarvam.ALL_LANGUAGES.get(s.language, s.language)
        script = SCRIPT_TEXT.get(s.language, f"Write in {lang_name} script; keep English business words in English.")
        hooks = "; ".join(h["value"] for h in plan["personalisation"]) or "none"
        playbook = "\n".join(f"- {p['objection']}: {p['approach']}" for p in plan["objection_playbook"])
        escalation = "\n".join(f"- When {e['when']}: {e['do']}" for e in plan["escalation_rules"])
        facts = "\n".join(f"- {v}" for v in sp["facts"].values())
        guard = "\n".join(f"- {g}" for g in plan["guardrails"])
        n_bot = sum(t.role == "bot" for t in self.turns)
        escalate = ("\nESCALATION NOW: apologise, offer a callback at a time they choose, and end the call politely."
                    if self.escalate else "")
        return f"""You are Payal, IndiaMART's voice assistant, on a live phone call with a seller. Goal: fix a short meeting between the seller and an IndiaMART sales executive.

FACTS (the only facts you may state):
{facts}

GUARDRAILS (never break these):
{guard}

SELLER (from IndiaMART's seller file): {sp['summary']}
Personalisation hooks: {hooks}

DELIVERY FOR THIS SELLER:
- Language: {lang_name}. {MIX_TEXT.get(s.mix, '')} {script}
- Formality: {s.formality.replace('_', ' ')}. Warmth: {s.warmth}. Empathy: {s.empathy}. Vocabulary: {s.vocabulary}.
- Length: at most {LENGTH_WORDS[s.sentence_length]} words per reply, 1-2 sentences.

ACTIVE PERSONA: {PERSONAS[self.persona_id].name}. {s.focus}{escalate}

OPENING PLAN: {' '.join(plan['opening']['steps'])}
OBJECTION PLAYBOOK:
{playbook}
ESCALATION RULES:
{escalation}

OUTPUT: only the words Payal speaks next, no stage directions, no markdown, no emoji. Write times as words. This is reply number {n_bot + 1}.
Most replies continue the call: do NOT add any tag to them.
Only on your final goodbye (meeting confirmed, seller refused per the rules, or asked not to be called), append ##HANGUP##.
If the seller has agreed a meeting time, append ##MEETING: <day and time>## to that reply."""

    def _messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt()}]
        for t in self.turns:
            msgs.append({"role": "assistant" if t.role == "bot" else "user", "content": t.text})
        if len(msgs) == 1:
            msgs.append({"role": "user", "content": "(The seller has picked up the call. Say your opening line.)"})
        # Restate the live settings last: models follow the latest instruction over the history's language.
        s = self.settings
        lang_name = sarvam.ALL_LANGUAGES.get(s.language, s.language)
        script = SCRIPT_TEXT.get(s.language, f"Use {lang_name} script.")
        msgs.append({"role": "system", "content": (
            f"Reply now in {lang_name} only, whatever language earlier turns used. {script} "
            f"At most {LENGTH_WORDS[s.sentence_length]} words. Speak as {PERSONAS[self.persona_id].name}: {s.focus}"
            + (" Escalation: apologise, offer a callback, end the call." if self.escalate else ""))})
        return msgs

    # ---- voice ----------------------------------------------------------------------------
    def speak(self, text: str) -> tuple[bytes, int]:
        """Synthesize sentence-by-sentence in parallel and join, to cut latency."""
        s = self.settings
        tts_lang = s.language if s.language in sarvam.TTS_LANGUAGES else "hi-IN"
        parts = [p for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()] or [text]
        t0 = time.perf_counter()
        futs = [POOL.submit(sarvam.synthesize, p, tts_lang, s.speaker, s.pace, s.expressiveness, 16000) for p in parts]
        wavs = [f.result()[0] for f in futs]
        return join_wavs(wavs), int((time.perf_counter() - t0) * 1000)

    def _reply(self) -> tuple[str, dict]:
        content, ms = sarvam.chat(self._messages(), max_tokens=220, temperature=0.4)
        meeting = re.search(r"##\s*MEETING:\s*([^<>#]+?)\s*##", content)
        if meeting and re.search(r"pending|unknown|tbd|day and time", meeting.group(1), re.I):
            meeting = None
        text = re.sub(r"##\s*(HANGUP|MEETING:.*?)\s*##|\[(END|MEETING:[^\]]*)\]", "", content)
        text = re.sub(r"\s+", " ", text).strip()
        # A goodbye never asks a question, and the opening never ends the call.
        last_seller = next((t for t in reversed(self.turns) if t.role == "seller"), None)
        may_end = (bool(meeting) or "meeting" in self.outcome or self.escalate
                   or (last_seller is not None and any(END_PATTERN.search(normalize(x or "")) for x in
                                                       (last_seller.text, last_seller.meta.get("romanized")))))
        end = "HANGUP" in content and may_end and not text.rstrip().endswith("?")
        if meeting:
            self.outcome["meeting"] = meeting.group(1).strip()
        if end:
            self.ended = True
        return text, {"llm_ms": ms, "end": end}

    # ---- turns ------------------------------------------------------------------------------
    def open(self) -> dict:
        text, meta = self._reply()
        audio, tts_ms = self.speak(text)
        self.turns.append(Turn("bot", text, {**meta, "tts_ms": tts_ms}))
        return {"reply": text, "audio": audio, "timings": {**meta, "tts_ms": tts_ms}, "switches": [],
                "settings": self.settings.as_dict(), "ended": self.ended, "persona": self.persona_card(),
                "start_reason": self.start_reason, "timeline": self.timeline}

    def persona_card(self) -> dict:
        p = PERSONAS[self.persona_id]
        return {"id": p.id, "name": p.name, "description": p.description, "colour": p.colour}

    def seller_audio(self, audio: bytes, mime: str = "audio/wav") -> dict:
        t0 = time.perf_counter()
        f_native = POOL.submit(sarvam.transcribe, audio, "transcribe", "unknown", "turn.wav", mime)
        f_roman = POOL.submit(sarvam.transcribe, audio, "translit", "unknown", "turn.wav", mime)
        native = f_native.result()
        try:
            roman = f_roman.result()
        except sarvam.SarvamError:
            roman = native
        stt_ms = int((time.perf_counter() - t0) * 1000)
        return self._seller_turn(native.text, roman.text, native.language_code, native.language_probability,
                                 {"stt_ms": stt_ms})

    def seller_text(self, text: str, language_code: str | None = None) -> dict:
        """Typed fallback (or tests): treats the text as both native and romanized."""
        return self._seller_turn(text, text, language_code, 1.0 if language_code else None, {"stt_ms": 0})

    def _seller_turn(self, native: str, roman: str, lang: str | None, prob: float | None, timings: dict) -> dict:
        if not native.strip():
            return {"transcript": "", "skipped": True, "switches": [], "settings": self.settings.as_dict()}
        turn_no = sum(t.role == "seller" for t in self.turns) + 1
        self.turns.append(Turn("seller", native, {"romanized": roman, "language": lang, "language_probability": prob}))
        t0 = time.perf_counter()
        convo = [(t.role, t.text) for t in self.turns]
        prev = self.verdicts[-1] if self.verdicts else None
        f_verdict = POOL.submit(analyse, convo, roman, self.persona_id, prev, self.use_llm_signals)
        ld = self.lang.update(lang, prob, [native, roman], len(native.split()))
        v = f_verdict.result()
        self.verdicts.append(v)
        decision = self.director.decide(v)
        timings["analyst_ms"] = int((time.perf_counter() - t0) * 1000)

        new_entries = []
        base_entry = {"call_id": self.id, "seller_turn": turn_no, "t_sec": round(time.time() - self.t0, 1),
                      "seller_said": native[:160]}
        if ld:
            changes, reason = apply_language(self.settings, self.base, ld)
            if ld.detail["to"] in sarvam.TTS_LANGUAGES:
                self.lang.current = self.settings.language
            if changes:
                new_entries.append({**base_entry, "type": "language", "signal": "language_switch",
                                    "detector": ld.detector, "score": round(ld.score, 2), "evidence": ld.evidence[:120],
                                    "from_persona": self.persona_id, "to_persona": self.persona_id,
                                    "change": changes, "reason": reason})
        if decision.action == "escalate":
            self.escalate = True
        if decision.switched:
            changes = apply_persona(self.settings, self.base, decision.to_persona)
            self.persona_id = decision.to_persona
            self.timeline.append({"seller_turn": turn_no, "persona": self.persona_id})
            new_entries.append({**base_entry, "type": "persona", "signal": v.state, "detector": v.source,
                                "score": round(v.intensity, 2), "trend": v.trend, "evidence": (v.evidence or native)[:120],
                                "from_persona": decision.from_persona, "to_persona": decision.to_persona,
                                "from_name": PERSONAS[decision.from_persona].name,
                                "to_name": PERSONAS[decision.to_persona].name,
                                "change": changes, "reason": decision.reason, "analyst_rationale": v.rationale})
        elif decision.action == "escalate":
            new_entries.append({**base_entry, "type": "escalation", "signal": v.state, "detector": v.source,
                                "score": round(v.intensity, 2), "trend": v.trend, "evidence": (v.evidence or native)[:120],
                                "from_persona": self.persona_id, "to_persona": self.persona_id,
                                "change": {"call": ["continue", "end with callback offer"]}, "reason": decision.reason})
        self.switch_log.extend(new_entries)

        t1 = time.perf_counter()
        text, meta = self._reply()
        timings["llm_ms"] = meta["llm_ms"]
        audio, timings["tts_ms"] = self.speak(text)
        timings["reply_total_ms"] = int((time.perf_counter() - t1) * 1000) + timings["stt_ms"] + timings["analyst_ms"]
        self.turns.append(Turn("bot", text, {"settings": self.settings.as_dict()}))
        self.save()
        return {"transcript": native, "romanized": roman, "language": lang, "language_probability": prob,
                "analysis": v.as_dict(), "decision": decision.__dict__, "switches": new_entries, "reply": text,
                "audio": audio, "settings": self.settings.as_dict(), "timings": timings, "ended": self.ended,
                "outcome": self.outcome, "persona": self.persona_card(), "timeline": self.timeline}

    def save(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        p = LOG_DIR / f"{self.id}.json"
        p.write_text(json.dumps({
            "call_id": self.id, "seller_id": self.spec.get("seller_id"), "seller_type": self.spec["seller_type"],
            "start_persona": self.timeline[0]["persona"], "start_reason": self.start_reason,
            "persona_timeline": self.timeline,
            "start_settings": self.base.as_dict(), "final_settings": self.settings.as_dict(),
            "switch_log": self.switch_log, "outcome": self.outcome,
            "analyst_verdicts": [v.as_dict() for v in self.verdicts],
            "transcript": [{"role": t.role, "text": t.text} for t in self.turns],
        }, indent=2, ensure_ascii=False, default=str))
        return p


def join_wavs(wavs: list[bytes], gap_ms: int = 120) -> bytes:
    if len(wavs) == 1:
        return wavs[0]
    params, frames = None, []
    for w in wavs:
        with wave.open(io.BytesIO(w)) as r:
            if params is None:
                params = r.getparams()
            frames.append(r.readframes(r.getnframes()))
    silence = b"\x00" * int(params.framerate * gap_ms / 1000) * params.sampwidth * params.nchannels
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(silence.join(frames))
    return out.getvalue()
