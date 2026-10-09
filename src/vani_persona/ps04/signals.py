"""Live seller-signal detection for the five PS04 signals.

Each seller utterance is scored by up to three detectors:

- **lexicon**: explicit phrases (romanized Hinglish, Devanagari, English), instant;
- **llm**: a Sarvam chat model scores all four behavioural signals in any language;
- **saaras**: the language Saaras detected in the audio (language switch only).

A signal fires when its fused score crosses its threshold. English <-> Hindi switches need
two consecutive utterances (sellers often drop an English sentence into Hinglish); a switch
into any other language, or an explicit request, fires immediately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from vani_persona.features import lexicons as lx
from vani_persona.features.transcript import normalize
from vani_persona.ps04 import sarvam

SIGNALS = ("frustration", "confusion", "interest", "rush", "language_switch")
THRESHOLDS = {"frustration": 0.6, "confusion": 0.6, "interest": 0.65, "rush": 0.6}
LEXICON_SCORE = 0.9
MIN_LANG_PROB = 0.6

FRUSTRATION = [
    "kitni baar", "kitne baar", "baar baar", "pareshan", "pareshaan", "irritate", "irritating", "dimag",
    "dimaag", "bakwas", "faltu", "faltu call", "bas karo", "band karo", "phone rakho", "rakh do",
    "time waste", "waste of time", "sar mat khao", "tang", "tang aa", "kya bakwas", "annoying",
    "stop it", "fed up", "har din call", "roz call", "gussa",
    "कितनी बार", "बार बार", "परेशान", "बकवास", "फालतू", "बस करो", "बंद करो", "दिमाग", "तंग",
]
INTEREST = [
    "kab aayenge", "kab aaoge", "kab aa sakte", "kitne baje", "kaise kaam karta", "aur batao", "aur bataiye",
    "batao batao", "haan batao", "haan bataiye", "interested", "interest hai", "chalega", "thik hai bhejo",
    "meeting rakh", "meeting kar", "aa jaiye", "aa jao", "kal aa", "visit kar", "sounds good",
    "tell me more", "how does it work", "when can", "what time", "fix kar do", "schedule kar",
    "कब आएंगे", "कितने बजे", "और बताइए", "हाँ बताइए", "मीटिंग रख", "आ जाइए", "चलेगा",
]
END_CALL = lx.OPT_OUT + lx.OBJECTION_NOT_INTERESTED
CONFUSION = lx.CLARIFICATION + ["kaun sa", "kya keh rahe", "kya bol rahe", "samajh nahi aaya", "kuch samajh nahi",
                                "क्या कह रहे", "समझ नहीं आया"]
RUSH = lx.BREVITY_REQUEST + lx.TIME_PRESSURE + ["2 minute", "do minute", "jaldi", "जल्दी"]


def _compile(phrases: list[str]) -> re.Pattern[str]:
    alts = sorted({re.escape(normalize(p)) for p in phrases if normalize(p)}, key=len, reverse=True)
    return re.compile(r"(?<![\wऀ-ॿ])(?:" + "|".join(alts) + r")(?![\wऀ-ॿ])")


PATTERNS = {"frustration": _compile(FRUSTRATION), "interest": _compile(INTEREST),
            "confusion": _compile(CONFUSION), "rush": _compile(RUSH)}
END_PATTERN = _compile(END_CALL)
LANG_REQUEST = {
    "en-IN": _compile(lx.LANGUAGE_REQUEST["en"]),
    "hi-IN": _compile(lx.LANGUAGE_REQUEST["hi"]),
    **{code: _compile([f"{name.lower()} mein", f"{name.lower()} me", f"speak in {name.lower()}",
                       f"{name.lower()} bolo", f"in {name.lower()}"])
       for code, name in sarvam.ALL_LANGUAGES.items() if code not in ("en-IN", "hi-IN")},
}

LLM_PROMPT = (
    "You score one seller utterance from a phone call between an IndiaMART voice agent and a seller. "
    "The utterance may be in any Indian language or code-mixed. Return only JSON with keys "
    '"frustration", "confusion", "interest", "rush" (each a number 0-1) and "evidence" (the seller words '
    "that show the strongest signal, max 12 words). Definitions: frustration = irritation, anger or annoyance; "
    "confusion = did not understand, asks what/why/who, asks to repeat; interest = wants details, asks how or when, "
    "agrees to a meeting or time; rush = busy, wants it quick, asks to get to the point or call later. "
    "Score 0 when absent. Do not reward politeness as interest. Asking who is calling or 'kaun?' at the start "
    "of a call is normal, not confusion. Asking to speak another language is not confusion by itself."
)


@dataclass
class Detection:
    signal: str
    score: float
    detector: str
    evidence: str
    detail: dict = field(default_factory=dict)


def lexicon_hits(texts: list[str]) -> dict[str, str]:
    hits = {}
    for sig, pat in PATTERNS.items():
        for t in texts:
            m = pat.search(normalize(t or ""))
            if m:
                hits[sig] = m.group(0)
                break
    return hits


def llm_scores(seller_text: str, last_bot_text: str) -> tuple[dict, int]:
    try:
        content, ms = sarvam.chat(
            [{"role": "system", "content": LLM_PROMPT},
             {"role": "user", "content": f"Agent said: {last_bot_text}\nSeller said: {seller_text}"}],
            max_tokens=120, temperature=0.0, json_mode=True)
        m = re.search(r"\{.*\}", content, re.S)
        out = json.loads(m.group(0)) if m else {}
        return {k: float(out.get(k, 0) or 0) for k in THRESHOLDS} | {"evidence": str(out.get("evidence", ""))}, ms
    except (sarvam.SarvamError, ValueError, TypeError):
        return {}, 0  # lexicons still run; a failed classifier never blocks the call


class LanguageTracker:
    """Hysteresis for language switches."""

    def __init__(self, current: str):
        self.current = current
        self.pending: str | None = None

    def update(self, detected: str | None, prob: float | None, texts: list[str], n_words: int) -> Detection | None:
        for code, pat in LANG_REQUEST.items():
            if code != self.current and any(pat.search(normalize(t or "")) for t in texts):
                self.pending = None
                return Detection("language_switch", 1.0, "lexicon:explicit request", texts[0],
                                 {"from": self.current, "to": code})
        if not detected or detected == self.current or n_words < 2 or (prob is not None and prob < MIN_LANG_PROB):
            self.pending = None
            return None
        pair = {detected, self.current} <= {"hi-IN", "en-IN"}
        if pair and self.pending != detected:
            self.pending = detected  # wait for a second utterance in the same language
            return None
        self.pending = None
        return Detection("language_switch", float(prob or 0.0),
                         f"saaras:{detected} p={prob:.2f}" + (" (2nd utterance)" if pair else ""),
                         texts[0], {"from": self.current, "to": detected})


def detect(native: str, romanized: str, last_bot_text: str, use_llm: bool = True) -> tuple[list[Detection], dict]:
    """Behavioural signals for one utterance. Returns detections above threshold + raw scores."""
    texts = [native, romanized]
    lex = lexicon_hits(texts)
    llm, llm_ms = llm_scores(native or romanized, last_bot_text) if use_llm else ({}, 0)
    raw = {"lexicon": lex, "llm": llm, "llm_ms": llm_ms}
    out = []
    for sig, thr in THRESHOLDS.items():
        lscore = LEXICON_SCORE if sig in lex else 0.0
        mscore = llm.get(sig, 0.0)
        score = max(lscore, mscore)
        if score >= thr:
            det = f"lexicon:'{lex[sig]}'" if lscore >= mscore else f"llm:{mscore:.2f}"
            if lscore and mscore >= thr:
                det = f"lexicon:'{lex[sig]}' + llm:{mscore:.2f}"
            out.append(Detection(sig, score, det, lex.get(sig) or llm.get("evidence", "") or native))
    # Interest and frustration together usually means sarcasm or a complaint: keep frustration.
    names = {d.signal for d in out}
    if {"interest", "frustration"} <= names:
        out = [d for d in out if d.signal != "interest"]
    return out, raw
