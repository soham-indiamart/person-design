"""Behaviour Analyst agent: reads the live transcript and reports the seller's state.

A separate Sarvam LLM call after every seller turn. It sees the last few turns of the
conversation (both sides), the persona VANI is using and its own previous verdict, so it
can tell escalation and calming apart, not just spot single phrases. Output:

    {state: calm|aggressive|rushed|confused|interested|suspicious,
     intensity: 0-1, trend: escalating|steady|calming, evidence, rationale}

Phrase lexicons run alongside it. They are the fallback when the LLM call fails, times
out or hits the credit cap, and a safety net: an explicit aggression phrase is never
overruled by a "calm" verdict.

    python -m vani_persona.ps04.analyst --selftest     # ~30 labelled turns, ~1 credit
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field

from vani_persona.features import lexicons as lx
from vani_persona.features.transcript import normalize
from vani_persona.ps04 import sarvam
from vani_persona.ps04.live_personas import PERSONAS, SELLER_STATES
from vani_persona.ps04.signals import CONFUSION, FRUSTRATION, INTEREST, RUSH, _compile, pace_request

WINDOW_TURNS = 6
TIMEOUT_S = 3.0

AGGRESSION = FRUSTRATION + [
    "dimaag kharab", "dimag kharab", "dimaag mat kha", "shut up", "chup", "bakwaas", "pagal", "bewakoof",
    "kya mazaak", "mazak hai", "harass", "harassment", "stop calling", "again and again", "complaint karunga", "police", "block kar", "nonsense", "rubbish",
    "दिमाग खराब", "पागल", "बेवकूफ", "मज़ाक", "शिकायत",
]
# Identity questions ("who is this", "kaun bol") are normal at call start, not suspicion.
SUSPICION = [p for p in lx.OBJECTION_TRUST if p not in {"who is this", "kaun bol", "kahan se"}] + [
    "bot ho", "robot", "machine ho", "recorded", "real person", "insaan ho", "asli", "otp", "how did you get my number",
    "बॉट", "रोबोट", "असली", "नंबर कहाँ से",
]
HESITANT = [
    "soch ke", "sochke", "soch kar", "sochta hoon", "sochenge", "dekhte hain", "dekhenge", "baad mein dekh",
    "abhi zaroorat nahi", "pata nahi", "shayad", "not sure", "let me think", "will think", "maybe later", "abhi nahi",
    "thoda time", "partner se baat", "ghar pe pooch", "सोच के", "सोचकर", "देखेंगे", "पता नहीं", "शायद",
]
COMPLAINING = [
    "koi enquiry nahi", "enquiry nahi aati", "enquiry nahi aayi", "leads nahi", "fake lead", "fake enquiry",
    "faltu leads", "faltu enquiry", "paise liye", "paisa barbaad", "paise doob", "koi fayda nahi", "fayda nahi hua",
    "executive nahi aaya", "koi nahi aaya", "service kharab", "bekar service", "bad experience", "no response",
    "cheating", "dhokha", "refund", "paisa wapas", "फायदा नहीं", "धोखा", "फर्जी लीड", "पैसे डूब",
]
INQUISITIVE = [
    "kya fayda", "kya benefit", "kaise kaam", "kaise hoga", "free hai", "paid hai", "charge", "kitna lagega",
    "kya karenge", "kya batayenge", "kis bare mein", "kis baare mein", "kya hai ye", "what is the benefit",
    "how does it work", "is it free", "what will", "details batao", "detail mein", "kya milega",
    "क्या फायदा", "कैसे काम", "फ्री है", "कितना लगेगा",
]
LEXICONS = {
    "aggressive": _compile(AGGRESSION), "suspicious": _compile(SUSPICION), "complaining": _compile(COMPLAINING),
    "confused": _compile(CONFUSION), "rushed": _compile(RUSH), "uninterested": _compile(lx.OBJECTION_NOT_INTERESTED),
    "hesitant": _compile(HESITANT), "inquisitive": _compile(INQUISITIVE), "interested": _compile(INTEREST),
}
PRIORITY = ["aggressive", "suspicious", "complaining", "confused", "rushed", "uninterested", "hesitant",
            "inquisitive", "interested", "calm"]

PROMPT = """You are a call-behaviour analyst watching a live phone call between Payal (IndiaMART's voice assistant) and a seller. Read the conversation and judge the seller's state RIGHT NOW, at their latest turn, in the context of the whole conversation.

States (pick one):
- calm: neutral, polite, answering normally, asking who is calling at the start.
- aggressive: irritated, angry, rude, annoyed by repeated calls, raising their voice in words (anger at THIS call).
- rushed: busy, wants it quick, asks to get to the point or call later.
- confused: did not understand, asks what or why, asks to repeat.
- interested: wants to go ahead: agrees to meet or proceed, proposes a time, gives details willingly.
- suspicious: doubts the caller is genuine (fraud, spam, bot, how did you get my number).
- hesitant: unsure, not refusing: wants to think, "baad mein dekhenge", not ready yet, needs to ask someone.
- complaining: unhappy about a PAST experience (no or fake enquiries, paid but no results, executive never came).
- inquisitive: evaluating with questions: what is the benefit, how does it work, free or paid, what exactly happens.
- uninterested: clearly says they do not want or need it (not interested, nahi chahiye).

Also give:
- intensity 0-1 (how strong; 0.3 mild, 0.6 clear, 0.9 extreme),
- trend versus their previous turns: escalating, steady or calming,
- evidence: the seller's words that show it (max 12 words, copy them),
- rationale: one short sentence.

Then profile the seller as a person, from the whole conversation so far:
- language: language of the seller's LATEST turn: hi-IN (Hindi or Hinglish, any script), en-IN (mostly full English sentences), bn-IN, gu-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, or another xx-IN code,
- mix: only for Hindi/Hinglish: hindi_heavy, balanced or english_heavy (how much English they mix in),
- pace: how fast the seller wants Payal to talk now: faster, slower or same,
- patience: low, medium or high,
- formality: formal (aap, ji, sir, complete sentences) or casual (tum, yaar, bhai, short),
- persona: 4-10 words describing who this seller seems to be right now (e.g. "busy shop owner, impatient, Hinglish").

Rules: judge the seller only, never Payal. Politeness alone is not interest. "Kaun?" or "who is this?" at the start is calm. If the seller was aggressive and now answers normally, that is calm with trend calming. Code-mixed Hinglish and any Indian language are normal. A complaint that Payal talks too slowly ("itna dheere kyu bol rahe ho", "too slow") is pace faster and state rushed, NOT confused. "Speak slowly", "aaram se boliye" is pace slower. Asking to speak another language is not confusion by itself.

Examples:
Seller: "haan ji bolo, kaun?" -> {"state":"calm","intensity":0.3,"trend":"steady","language":"hi-IN","mix":"hindi_heavy","pace":"same","patience":"medium","formality":"formal"}
Seller: "kitni baar call karoge yaar, dimaag kharab kar diya" -> {"state":"aggressive","intensity":0.85,"trend":"escalating","patience":"low","formality":"casual"}
Seller (after being angry): "achha theek hai, bolo kya hai" -> {"state":"calm","intensity":0.4,"trend":"calming"}
Seller: "fraud call hai kya? number kahan se mila?" -> {"state":"suspicious","intensity":0.75,"trend":"escalating"}
Seller: "bhai itna dheere kyu bol rahe ho, jaldi bolo" -> {"state":"rushed","intensity":0.75,"trend":"escalating","pace":"faster","patience":"low","formality":"casual"}
Seller: "kal kitne baje aa sakte ho?" -> {"state":"interested","intensity":0.8,"trend":"steady"}
Seller: "pehle bhi paise diye the, ek bhi achhi enquiry nahi aayi" -> {"state":"complaining","intensity":0.75,"trend":"escalating"}
Seller: "ye meeting free hai ya paid? isse kya fayda hoga?" -> {"state":"inquisitive","intensity":0.7,"trend":"steady"}
Seller: "dekhte hain, abhi soch ke batata hoon" -> {"state":"hesitant","intensity":0.65,"trend":"steady"}
Seller: "nahi ji, humein nahi chahiye" -> {"state":"uninterested","intensity":0.8,"trend":"steady"}
Seller: "Okay, tell me what this meeting is about" -> {"state":"calm","intensity":0.4,"language":"en-IN","pace":"same"}

Return only JSON: {"state": "...", "intensity": 0.0, "trend": "...", "evidence": "...", "rationale": "...", "language": "...", "mix": "...", "pace": "...", "patience": "...", "formality": "...", "persona": "..."}"""

PROFILE_VALUES = {
    "mix": {"hindi_heavy", "balanced", "english_heavy"},
    "pace": {"faster", "slower", "same"},
    "patience": {"low", "medium", "high"},
    "formality": {"formal", "casual"},
}


@dataclass
class Verdict:
    state: str
    intensity: float
    trend: str
    evidence: str
    rationale: str
    source: str            # analyst | lexicon | analyst+lexicon
    lexicon: dict
    ms: int = 0
    profile: dict = field(default_factory=dict)   # language, mix, pace, patience, formality, persona

    def as_dict(self) -> dict:
        return asdict(self)


def _profile(o: dict) -> dict:
    out = {k: str(o[k]).strip().lower() for k in PROFILE_VALUES
           if str(o.get(k, "")).strip().lower() in PROFILE_VALUES[k]}
    lang = str(o.get("language", "")).strip()
    if re.fullmatch(r"[a-z]{2,3}-IN", lang):
        out["language"] = lang
    if o.get("persona"):
        out["persona"] = str(o["persona"])[:120]
    return out


def lexicon_states(texts: list[str]) -> dict[str, str]:
    hits = {}
    for state, pat in LEXICONS.items():
        for t in texts:
            m = pat.search(normalize(t or ""))
            if m:
                hits[state] = m.group(0)
                break
    return hits


def _lexicon_verdict(hits: dict, note: str) -> Verdict:
    for state in PRIORITY:
        if state in hits:
            return Verdict(state, 0.8, "steady", hits[state], f"Phrase '{hits[state]}' ({note}).", "lexicon", hits)
    return Verdict("calm", 0.3, "steady", "", f"No phrase matched ({note}).", "lexicon", hits)


def _format(turns: list[tuple[str, str]], who: str = "Seller") -> str:
    return "\n".join(f"{'Payal' if r == 'bot' else who}: {t}" for r, t in turns[-WINDOW_TURNS:])


def _prompt(counterpart: str) -> str:
    if counterpart == "seller":
        return PROMPT
    return (PROMPT.replace("seller", counterpart).replace("Seller", counterpart.capitalize())
            + f"\nThe {counterpart} called IndiaMART's help desk; 'meeting' examples apply to connecting with a seller.")


def analyse(turns: list[tuple[str, str]], romanized_latest: str, persona_id: str,
            previous: Verdict | None = None, use_llm: bool = True, counterpart: str = "seller") -> Verdict:
    """``turns``: (role, text) pairs ending with the other party's latest turn."""
    latest = turns[-1][1] if turns else ""
    hits = lexicon_states([latest, romanized_latest])
    want, phrase = pace_request([latest, romanized_latest])
    if want == "faster":
        # "itna dheere bol rahe ho, samajh nahi aa raha": a speed complaint, not confusion.
        hits.pop("confused", None)
        hits.setdefault("rushed", phrase)
    if not use_llm:
        return _lexicon_verdict(hits, "LLM analyst off")
    prev = (f"Your previous verdict: {previous.state} (intensity {previous.intensity:.1f}, {previous.trend})."
            if previous else "No previous verdict (first seller turn).")
    user = (f"Payal is currently speaking as: {PERSONAS[persona_id].name}.\n{prev}\n\n"
            f"Conversation (latest last):\n{_format(turns, counterpart.capitalize())}")
    t0 = time.perf_counter()
    try:
        content, _ = sarvam.chat([{"role": "system", "content": _prompt(counterpart)},
                                  {"role": "user", "content": user}],
                                 max_tokens=240, temperature=0.0, json_mode=True, timeout=TIMEOUT_S, retries=0)
        m = re.search(r"\{.*\}", content, re.S)
        o = json.loads(m.group(0)) if m else {}
        state = str(o.get("state", "")).lower().strip()
        if state not in SELLER_STATES:
            raise ValueError(f"bad state {state!r}")
        v = Verdict(state, max(0.0, min(1.0, float(o.get("intensity", 0.5) or 0.5))),
                    str(o.get("trend", "steady")).lower() if str(o.get("trend", "")).lower() in
                    {"escalating", "steady", "calming"} else "steady",
                    str(o.get("evidence", ""))[:120], str(o.get("rationale", ""))[:200], "analyst", hits,
                    profile=_profile(o))
    except (sarvam.SarvamError, ValueError, TypeError) as e:
        v = _lexicon_verdict(hits, f"analyst unavailable: {type(e).__name__}")
    v.ms = int((time.perf_counter() - t0) * 1000)
    if v.source == "analyst":
        if "aggressive" in hits and v.state != "aggressive":
            # Safety net: an explicit aggression phrase wins over a softer reading.
            v = Verdict("aggressive", max(0.7, v.intensity), "escalating", hits["aggressive"],
                        f"Aggression phrase '{hits['aggressive']}' overrides analyst '{v.state}'.",
                        "lexicon", hits, v.ms, v.profile)
        elif want == "faster" and v.state == "confused":
            v.state, v.source = "rushed", "analyst+lexicon"
            v.rationale = f"Speed complaint '{phrase}': wants it faster, not re-explained. {v.rationale}"[:200]
        elif v.state in hits:
            v.source = "analyst+lexicon"
    return v


SELFTEST = [
    ("calm", "हाँ जी, बोलिए, कौन?"), ("calm", "haan ji batao"), ("calm", "Hello, yes, who is this?"),
    ("calm", "ji main hi bol raha hoon, boliye"), ("calm", "achha, theek hai, aage bataiye"),
    ("aggressive", "kitni baar call karoge yaar, pareshan kar diya hai"),
    ("aggressive", "dimaag kharab mat karo, bakwas band karo"),
    ("aggressive", "Stop calling me again and again, this is harassment"),
    ("aggressive", "बार बार फोन क्यों करते हो, तंग आ गया हूँ"),
    ("aggressive", "tumhara roz ka yahi drama hai, complaint karunga"),
    ("rushed", "abhi busy hoon, jaldi bolo"), ("rushed", "dukaan pe customer hai, short mein batao"),
    ("rushed", "I'm driving, make it quick"), ("rushed", "baad mein call karna, abhi meeting mein hoon"),
    ("confused", "samajh nahi aaya, kaun si meeting?"), ("confused", "matlab kya hai iska?"),
    ("confused", "Sorry, I didn't get you, what is this about?"), ("confused", "क्या बोला आपने? फिर से बताइए"),
    ("interested", "kal kitne baje aa sakte hain?"), ("inquisitive", "achha, kaise kaam karta hai ye, detail mein batao"),
    ("inquisitive", "meeting free hai ya paid? isse mujhe kya fayda hoga?"),
    ("hesitant", "dekhte hain, abhi soch ke batata hoon"), ("hesitant", "partner se baat karke bataunga"),
    ("complaining", "pichhli baar paise diye the, ek bhi achhi enquiry nahi aayi"),
    ("complaining", "aapka executive aane wala tha, koi nahi aaya"),
    ("uninterested", "nahi ji, humein nahi chahiye"), ("uninterested", "Not interested, thank you"),
    ("interested", "Sounds good, when can your executive come?"), ("interested", "theek hai meeting rakh lo, 12 baje"),
    ("suspicious", "fraud call hai kya? number kahan se mila?"), ("suspicious", "aap bot ho kya? real person se baat karao"),
    ("suspicious", "How do I know this is really IndiaMART?"), ("suspicious", "ye spam to nahi hai?"),
    ("calm", "nahi nahi, main sun raha hoon, bolo"), ("rushed", "2 minute mein batao jo bhi hai"),
    ("aggressive", "pagal samjha hai kya, phone rakho"), ("interested", "haan bhejo executive ko, Surat mein hoon"),
]


def selftest(use_llm: bool = True) -> None:
    import collections
    conf = collections.Counter()
    ok = 0
    opener = ("bot", "Namaste, main IndiaMART se Payal bol rahi hoon. Aapke business ke liye ek chhoti meeting rakhni thi.")
    for gold, text in SELFTEST:
        v = analyse([opener, ("seller", text)], text, "normal", use_llm=use_llm)
        conf[(gold, v.state)] += 1
        ok += v.state == gold
        if v.state != gold:
            print(f"  miss: gold={gold:<11} got={v.state:<11} ({v.source}) {text[:60]}")
    print(f"\naccuracy {ok}/{len(SELFTEST)} = {ok / len(SELFTEST):.0%} ({'analyst' if use_llm else 'lexicon only'})")
    states = list(PRIORITY)
    print("gold \\ predicted | " + " | ".join(s[:5] for s in states))
    for g in states:
        print(f"{g:<16} | " + " | ".join(f"{conf[(g, p)]:>5}" for p in states))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lexicon-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(use_llm=not args.lexicon_only)
        print(json.dumps(sarvam.BUDGET.status(), indent=2) if not args.lexicon_only else "")


if __name__ == "__main__":
    main()
