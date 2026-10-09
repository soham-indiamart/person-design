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
    return re.compile(r"(?<![\w\u0900-\u0DFF])(?:" + "|".join(alts) + r")(?![\w\u0900-\u0DFF])")


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

# Requests about how fast Payal talks. "Itna dheere kyu bol rahe ho" asks for FASTER speech,
# so the faster list is checked first and is longer-match-first.
PACE_FASTER = [
    "itna dheere", "itna dhire", "kitna dheere", "dheere kyu", "dheere kyon", "dhire kyu", "dheere kyun",
    "bahut dheere", "bahut slow", "itna slow", "kitna slow", "too slow", "so slow", "very slow", "slow bol",
    "slow mat", "speak fast", "speak faster", "talk faster", "faster please", "be quick", "jaldi jaldi bolo",
    "tez bolo", "tez boliye", "fast bolo", "fast boliye", "thoda fast", "thoda tez", "jaldi bolo", "jaldi boliye",
    "इतना धीरे", "कितना धीरे", "धीरे क्यों", "बहुत धीरे", "इतना स्लो", "तेज़ बोलो", "तेज बोलो", "जल्दी बोलो",
    "এত স্লো", "এত আস্তে",
]
PACE_SLOWER = [
    "dheere bolo", "dheere boliye", "dhire bolo", "dhire boliye", "aaram se bolo", "aaram se boliye",
    "slowly", "slow down", "speak slow", "too fast", "so fast", "bahut fast", "itna fast", "itna tez",
    "thoda dheere", "thoda slow", "dheere dheere", "जल्दी जल्दी मत", "धीरे बोलो", "धीरे बोलिए", "आराम से",
    "इतना तेज़", "बहुत तेज़", "আস্তে বলুন",
]
PACE_PATTERNS = {"faster": _compile(PACE_FASTER), "slower": _compile(PACE_SLOWER)}

# Unicode blocks of the Indic scripts Bulbul can speak; Devanagari and Latin are ambiguous
# (Hindi / Marathi, English / romanized Hindi) and are left to Saaras and the analyst.
SCRIPT_LANG = [
    ((0x0980, 0x09FF), "bn-IN"), ((0x0A00, 0x0A7F), "pa-IN"), ((0x0A80, 0x0AFF), "gu-IN"),
    ((0x0B00, 0x0B7F), "od-IN"), ((0x0B80, 0x0BFF), "ta-IN"), ((0x0C00, 0x0C7F), "te-IN"),
    ((0x0C80, 0x0CFF), "kn-IN"), ((0x0D00, 0x0D7F), "ml-IN"),
]
DEVANAGARI = (0x0900, 0x097F)


# A turn that ends on one of these is usually unfinished: the speaker paused to think or
# stammered ("main kehna chah raha tha ki...", "aur...", "umm", "because").
TRAILING_CONNECTORS = {
    "aur", "or", "to", "toh", "ki", "ke", "ka", "matlab", "yaani", "yani", "jaise", "lekin", "magar", "kyunki",
    "kyonki", "kyuki", "agar", "jo", "wo", "woh", "vo", "voh", "phir", "fir", "ya", "aa", "um", "umm", "uh", "uhh",
    "uhm", "er", "erm", "and", "but", "because", "the", "a", "an", "of", "with", "like", "if", "or", "i", "my",
    "basically", "actually", "और", "तो", "कि", "के", "का", "लेकिन", "क्योंकि", "मतलब", "यानी", "अगर", "जो", "वो", "फिर", "या",
}
FILLERS = {"aa", "um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "matlab", "wo", "woh", "vo", "voh", "ki", "toh",
           "to", "haan", "ji", "मतलब", "वो"}
TRAILING_PUNCT = ("...", "…", ",", "-", "—")


def looks_incomplete(text: str) -> tuple[bool, str]:
    """(True, why) if the utterance sounds cut off mid-thought, so the agent should keep listening."""
    raw = (text or "").strip()
    if not raw:
        return False, ""
    if raw.endswith(TRAILING_PUNCT):
        return True, "trails off"
    words = normalize(raw).replace("?", " ").split()
    if not words:
        return False, ""
    if raw.endswith("?"):
        return False, ""
    if all(w in FILLERS for w in words) and any(w not in {"haan", "ji", "hmm"} for w in words):
        return True, "only fillers so far"
    if words[-1] in TRAILING_CONNECTORS and not (len(words) == 1 and words[0] in {"to", "toh", "ji"}):
        return True, f"ends on '{words[-1]}'"
    return False, ""


def pace_request(texts: list[str]) -> tuple[str | None, str]:
    """('faster' | 'slower' | None, matched phrase)."""
    for want in ("faster", "slower"):
        for t in texts:
            m = PACE_PATTERNS[want].search(normalize(t or ""))
            if m:
                return want, m.group(0)
    return None, ""


def script_language(text: str) -> tuple[str | None, float]:
    """Language implied by the dominant non-Devanagari Indic script, with its share of letters."""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        letters += 1
        o = ord(ch)
        for (lo, hi), code in SCRIPT_LANG:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts or not letters:
        return None, 0.0
    code = max(counts, key=counts.get)
    return code, counts[code] / letters


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
    """Decides when VANI's reply language follows the seller.

    Evidence, strongest first: an explicit request ("Bangla mein bolo"); the script of the
    transcript (Bengali, Tamil, ... letters); Saaras's detected language; the analyst's
    reading of the turn (``hint``). A full sentence in a new language switches at once;
    a short English fragment inside Hinglish (or vice versa) needs a second utterance.
    """

    MIN_FULL_SENTENCE = 4
    SCRIPT_SHARE = 0.6
    SURE_PROB = 0.8

    def __init__(self, current: str):
        self.current = current
        self.pending: str | None = None

    def _decide(self, detected: str | None, prob: float | None, texts: list[str], n_words: int,
                hint: str | None) -> tuple[Detection | None, str | None]:
        """Returns (detection or None, new pending). Pure: does not change the tracker."""
        for code, pat in LANG_REQUEST.items():
            if code != self.current and any(pat.search(normalize(t or "")) for t in texts):
                return Detection("language_switch", 1.0, "lexicon:explicit request", texts[0],
                                 {"from": self.current, "to": code}), None
        script, share = script_language(texts[0] if texts else "")
        if script and script != self.current and share >= self.SCRIPT_SHARE and n_words >= 2:
            to = detected if detected in ("as-IN",) and script == "bn-IN" else script
            return Detection("language_switch", round(share, 2), f"script:{to} {share:.0%} of letters",
                             texts[0], {"from": self.current, "to": to}), None
        saaras = detected if detected and (prob is None or prob >= MIN_LANG_PROB) else None
        candidate = saaras if saaras and saaras != self.current else hint if hint and hint != self.current else None
        if not candidate or n_words < 2:
            return None, None
        agree = saaras == candidate and hint == candidate
        pair = {candidate, self.current} <= {"hi-IN", "en-IN"}
        sure = (agree or (saaras == candidate and (prob or 0) >= self.SURE_PROB)
                or (saaras is None and hint == candidate))
        immediate = not pair or (n_words >= self.MIN_FULL_SENTENCE and sure) or self.pending == candidate
        if not immediate:
            return None, candidate  # wait for a second utterance in the same language
        how = ("saaras+analyst" if agree else f"saaras:{candidate}" if saaras == candidate else f"analyst:{candidate}")
        detail = f"{how} p={prob:.2f}" if prob is not None and saaras == candidate else how
        if pair and self.pending == candidate:
            detail += " (2nd utterance)"
        return Detection("language_switch", float(prob if prob is not None else 0.8), detail, texts[0],
                         {"from": self.current, "to": candidate}), None

    def peek(self, detected: str | None, prob: float | None, texts: list[str], n_words: int,
             hint: str | None = None) -> str:
        """Language the reply would use, without committing (for the speculative reply)."""
        d, _ = self._decide(detected, prob, texts, n_words, hint)
        return d.detail["to"] if d else self.current

    def update(self, detected: str | None, prob: float | None, texts: list[str], n_words: int,
               hint: str | None = None) -> Detection | None:
        d, self.pending = self._decide(detected, prob, texts, n_words, hint)
        return d


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
