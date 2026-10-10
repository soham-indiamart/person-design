"""Live call engine: seller audio -> Behaviour Analyst -> Persona Director -> reply -> Sarvam voice.

The seller spec (generator) sets the baseline: speaker, language, formality, plan, and
the starting persona (or the operator picks one). After every seller turn the Behaviour
Analyst agent reads the transcript, reports the seller's state and updates the live
seller profile (``profile.py``: language, pace wanted, patience, formality). The Persona
Director decides whether to switch to another named live persona (``live_personas.py``);
the profile then adjusts delivery on top of it. Language mirroring runs alongside,
independent of persona. Every switch is written to the switch log: the signal, the
detector and the seller's words, the change made, and why. The speaker, facts,
guardrails and call purpose never change.

Latency: the reply is drafted speculatively, in parallel with the analyst, using the
current persona; it is kept when the analyst's verdict leaves the text-shaping settings
unchanged. Speech is synthesised sentence by sentence so the first sentence can play
while the rest is still being generated (``speech_chunks``).
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import io
import json
import random
import re
import threading
from collections import Counter
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
from vani_persona.ps04.flows import HUMAN_TOUCH, flow_card, flow_for
from vani_persona.ps04.live_personas import PERSONAS, starting_persona
from vani_persona.ps04.profile import LiveProfile, apply_profile, profile_from_settings
from vani_persona.ps04.signals import (END_PATTERN, Detection, LanguageTracker, looks_incomplete, pace_request,
                                       wants_to_end)

MEETING_MODES = ("seller", "win_back")   # flows that end in a booked meeting (##MEETING## tag)

MAX_WAITS = 2      # consecutive "sounds unfinished, keep listening" holds before replying anyway
MAX_NUDGES = 2     # "are you still there?" check-ins before closing the call

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
    "en-IN": "Write in English (Latin script) only: no words or script from any other language.",
}
LENGTH_WORDS = {"normal": 28, "short": 18, "very_short": 12}

# How a person talks on the phone, for every journey (the flow's own style lines come on top).
HUMAN_SPEECH = (
    "Talk the way people talk on a phone call, not the way they write: short spoken sentences, the everyday spoken "
    "form of the language you are speaking (in Hindi: achha, dekhiye, haan toh, matlab, bas), English business "
    "words in English.",
    "Never use bookish Hindi: say shukriya / thank you, not dhanyavad; madad / help, not sahayata; "
    "'koi baat nahi', not 'koi samasya nahi'; never 'svabhavik', 'khed', 'kripya', 'avashya', 'upalabdh'.",
    "Do not follow one template every time (acknowledge + repeat + question). Sometimes just answer, sometimes just "
    "react and ask, sometimes a 4-6 word reply is enough. Vary the length.",
    "Never restate what they just said. React to it like a person would in two or three words "
    "('arre wah', 'ohh, ye toh dikkat hai', 'haan, samajh sakti hoon') and move on.",
    "Use their name, or sir / madam, at most once every three replies. Say 'note kar liya' at most once per call.",
    "Apologise at most once in the whole call, and only for something they actually told you went wrong.",
    "Accept approximate answers ('do-teen din' is fine); never ask a busy person to be more exact.",
    "The example phrases here are romanised only for reference: always write in the script given under DELIVERY.",
    "Ask questions the way people do: 'batayiye...', 'aap ... karte ho na?', not 'kya aap mujhe bata sakte hain'.",
    "If you do not know something, say so simply ('ye mujhe abhi pakka nahi pata, executive bata denge').",
)
# One instant spoken acknowledgement while the reply is being written, so there is no dead air.
ACKS = {"hi-IN": ("अच्छा।", "जी, अच्छा।", "अच्छा जी।", "हाँ जी।", "ठीक है जी।"),
        "en-IN": ("Okay.", "Right.", "I see.", "Got it.", "Okay, sure."),
        "ta-IN": ("சரி.", "சரி, சரி."), "te-IN": ("సరే.", "అలాగే."),
        "kn-IN": ("ಸರಿ.", "ಸರಿ, ಸರಿ."), "ml-IN": ("ശരി.", "ശരി, ശരി.")}
ACK_MIN_WORDS = 4
# Said if the model's goodbye is empty or still asks something.
GOODBYE_NOTE = {
    "hi-IN": "आपका बहुत-बहुत धन्यवाद जी, आपने समय दिया। कभी भी ज़रूरत हो तो IndiaMART हाज़िर है। नमस्ते!",
    "en-IN": "Thank you so much for your time. IndiaMART is always here whenever you need us. Have a great day!",
    "ta-IN": "உங்கள் நேரத்திற்கு மிக்க நன்றி. எப்போது தேவைப்பட்டாலும் IndiaMART உங்களுடன் இருக்கும். வணக்கம்!",
    "te-IN": "మీ సమయానికి చాలా ధన్యవాదాలు. ఎప్పుడు అవసరమైనా IndiaMART మీకు అందుబాటులో ఉంటుంది. నమస్కారం!",
    "kn-IN": "ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ತುಂಬಾ ಧನ್ಯವಾದಗಳು. ಯಾವಾಗ ಬೇಕಾದರೂ IndiaMART ನಿಮ್ಮೊಂದಿಗಿದೆ. ನಮಸ್ಕಾರ!",
    "ml-IN": "നിങ്ങളുടെ സമയത്തിന് വളരെ നന്ദി. എപ്പോൾ ആവശ്യമുണ്ടെങ്കിലും IndiaMART കൂടെയുണ്ട്. നമസ്കാരം!",
}          # "haan" / "ok" get no acknowledgement: the reply itself follows at once
ACK_CACHE: dict[tuple, bytes] = {}
# Pure acknowledgements only: a leading "haan" / "bilkul" can be the answer itself ("yes, it is free").
LEADING_ACK = re.compile(r"^\s*((जी|अच्छा|ठीक है|ji|achha|acha|accha|theek hai|okay|ok|right)(?=[\s,।.!])[\s,।.!]*){1,2}",
                         re.I)


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
    s = LiveSettings(
        language=lang["primary"]["value"], mix=lang["hinglish_mix"]["value"], formality=lang["formality"]["value"],
        speaker=v["speaker"]["value"], pace=float(v["pace"]["value"]), expressiveness=float(v["expressiveness"]["value"]),
        warmth=tone["warmth"]["value"], empathy=tone["empathy"]["value"],
        sentence_length="short" if spec["seller_type"] == "time_pressed" else "normal",
    )
    for k, d in (spec.get("use_case") or {}).get("delivery", {}).items():
        setattr(s, k, d["value"])
    return s


DELIVERY_KEYS = ("pace", "expressiveness", "warmth", "empathy", "sentence_length", "vocabulary")
# Settings that shape the reply text; pace and expressiveness only affect the voice.
TEXT_KEYS = ("language", "mix", "formality", "sentence_length", "vocabulary", "focus")


def apply_persona(s: LiveSettings, base: LiveSettings, persona_id: str, mode: str = "seller") -> dict:
    """Set the delivery settings of ``persona_id`` (seller baseline + persona template).

    Language, mix, formality and speaker are left alone. Returns {setting: [old, new]}.
    """
    before = s.as_dict()
    template = PERSONAS[persona_id].delivery
    for k in DELIVERY_KEYS:
        setattr(s, k, template.get(k, getattr(base, k)))
    s.focus = PERSONAS[persona_id].behaviour_for(mode)
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


def mix_for(language: str, analyst_mix: str | None, current_mix: str, base_mix: str) -> str:
    if language == "en-IN":
        return "english_heavy"
    if language != "hi-IN":
        return "native"
    if analyst_mix:
        return analyst_mix
    return current_mix if current_mix in ("hindi_heavy", "balanced", "english_heavy") else base_mix


def trim_rambling(text: str, limit: int) -> str:
    """A person on the phone does not deliver a paragraph: if a reply runs far past its length, drop the
    opening reaction and keep the last two sentences (the point and the question that refers to it)."""
    parts = [p for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]
    if len(text.split()) <= limit * 1.5 or len(parts) < 3:
        return text
    return " ".join(parts[-2:])


# Things a person says once per call; later repeats are cut out of the reply.
ONCE_PER_CALL = {
    "apology": re.compile(r"माफ़ी|माफी|माफ़ कीजिए|क्षमा|खेद|maafi|maaf kijiye|sorry|apolog", re.I),
    "noted": re.compile(r"note\s*(कर|kar)|नोट\s*कर", re.I),
}


def drop_repeats_once(text: str, said_before: str) -> str:
    """Remove clauses that apologise / say 'noted' again when Payal already did earlier in the call."""
    kinds = [k for k, rx in ONCE_PER_CALL.items() if rx.search(said_before)]
    if not kinds:
        return text
    clauses = re.split(r"(?<=[,.!?।])\s+", text)
    kept = [c for c in clauses if not any(ONCE_PER_CALL[k].search(c) for k in kinds)]
    if not kept or kept == clauses:
        return text
    out = " ".join(kept).strip()
    return re.sub(r",\s*$", "।" if re.search(r"[\u0900-\u097F]", out) else ".", out)


MARKED_TAG = re.compile(r"##\s*(?:NOTE|STAGE|MEETING|HANGUP)\b[^#]*##")
UNMARKED_TAG = re.compile(r"#{0,2}\s?\b(NOTE|STAGE|MEETING)\s*:\s*([^\n#\x00]+)#*")


def mark_tags(content: str) -> str:
    """Wrap tags the model wrote without (or with half) their ## markers, so they are never spoken."""
    kept: list[str] = []

    def keep(m: re.Match) -> str:
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"
    content = MARKED_TAG.sub(keep, content)
    content = UNMARKED_TAG.sub(lambda m: f" ##{m.group(1)}: {m.group(2).strip()}##", content)
    content = re.sub(r"(?<!#)\bHANGUP\b", "##HANGUP##", content)
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], content)


def split_speech(text: str) -> list[str]:
    """Sentences for TTS. A long first sentence is cut at its first comma so audio starts sooner."""
    parts = [p for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()] or [text]
    first = parts[0]
    if len(first.split()) > 10:
        m = re.search(r"^(\S+(?:\s+\S+){2,}?[,،])\s+(.+)$", first)
        if m and len(m.group(2).split()) >= 3:
            parts[0:1] = [m.group(1), m.group(2)]
    return parts


class CallSession:
    def __init__(self, spec: dict, use_llm_signals: bool = True, start_persona: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.spec = spec
        self.flow = flow_for(spec)
        self.flow_state: dict = {"stage": self.flow.stages[0].id, "visited": [self.flow.stages[0].id], "notes": {}}
        self.pending: dict | None = None   # an unfinished utterance held while the caller keeps talking
        self.waits = 0
        self.nudges = 0
        self.lock = threading.Lock()
        self.base = settings_from_spec(spec)
        if start_persona in PERSONAS:
            self.persona_id = start_persona
            self.start_reason = f"Chosen before the call: {PERSONAS[start_persona].name}. Switches automatically from here."
        else:
            self.persona_id, self.start_reason = starting_persona(spec)
        # Persona to settle back into when the seller is calm: what the use case and archetype call for.
        self.rest_persona = (spec.get("use_case") or {}).get("start_persona") or "normal"
        # These speak at the default pace; a switch away to another persona brings in its faster / slower pace.
        self.home_personas = {self.persona_id, self.rest_persona}
        self.closing: str | None = None      # why the call is being wrapped up, once the caller signals it
        self.profile: LiveProfile = profile_from_settings(self.base.language, self.base.mix, self.base.formality,
                                                          spec.get("seller_type", ""))
        self.unvoiced: str | None = None  # a language Saaras understands but Bulbul cannot speak
        self.settings = self.compose(self.persona_id, self.profile)
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
        self.ack: str | None = None          # the acknowledgement spoken while this reply is generated
        self.recent_acks: list[str] = []

    # ---- instant acknowledgement ------------------------------------------------------------
    def warm_acks(self) -> None:
        """Synthesise the acknowledgements for the current voice in the background (cached across calls)."""
        lang, speaker, pace, expr, sr = self._tts_args(self.settings)
        for text in ACKS.get(lang, ()):
            key = (text, lang, speaker)
            if key not in ACK_CACHE:
                ACK_CACHE[key] = b""         # claimed: synthesis in flight
                POOL.submit(self._cache_ack, key, (lang, speaker, pace, expr, sr))

    @staticmethod
    def _cache_ack(key: tuple, args: tuple) -> None:
        try:
            ACK_CACHE[key] = sarvam.synthesize(key[0], *args)[0]
        except Exception:
            ACK_CACHE.pop(key, None)

    def pick_ack(self, text: str) -> tuple[str, bytes] | None:
        """A short 'achha' / 'ji' to say at once, like a person, unless it would sound wrong here."""
        if self.escalate or self.persona_id == "de_escalator" or len(text.split()) < ACK_MIN_WORDS:
            return None
        lang, speaker = self._tts_args(self.settings)[:2]
        options = [t for t in ACKS.get(lang, ()) if ACK_CACHE.get((t, lang, speaker)) and t not in self.recent_acks[-2:]]
        if not options:
            return None
        t = random.choice(options)
        self.recent_acks.append(t)
        return t, ACK_CACHE[(t, lang, speaker)]

    # ---- settings ---------------------------------------------------------------------
    def compose(self, persona_id: str, profile: LiveProfile) -> LiveSettings:
        """Seller baseline -> live language -> persona delivery -> live profile adjustments."""
        s = copy.deepcopy(self.base)
        s.language, s.mix = profile.language, profile.mix
        apply_persona(s, self.base, persona_id, self.flow.mode)
        if persona_id in self.home_personas:
            s.pace = self.base.pace
        apply_profile(s, profile)
        if self.unvoiced:
            s.mix, s.vocabulary = "hindi_heavy", "simple"
            s.focus += (f" The seller speaks {sarvam.ALL_LANGUAGES.get(self.unvoiced, self.unvoiced)}, which you "
                        "cannot speak: use simple Hindi and offer an executive who speaks their language.")
        return s

    # ---- prompt -----------------------------------------------------------------------
    def system_prompt(self, s: LiveSettings | None = None, persona_id: str | None = None,
                      escalate: bool | None = None) -> str:
        sp = self.spec
        s = s or self.settings
        persona_id = persona_id or self.persona_id
        escalate = self.escalate if escalate is None else escalate
        plan = sp["conversation_plan"]
        lang_name = sarvam.ALL_LANGUAGES.get(s.language, s.language)
        script = SCRIPT_TEXT.get(s.language, f"Write in {lang_name} script; keep English business words in English.")
        hooks = "; ".join(h["value"] for h in plan["personalisation"]) or "none"
        playbook = "\n".join(f"- {p['objection']}: {p['approach']}" for p in plan["objection_playbook"])
        escalation = "\n".join(f"- When {e['when']}: {e['do']}" for e in plan["escalation_rules"])
        facts = "\n".join(f"- {v}" for v in sp["facts"].values())
        guard = "\n".join(f"- {g}" for g in plan["guardrails"])
        n_bot = sum(t.role == "bot" for t in self.turns)
        esc = ("\nESCALATION NOW: apologise, offer a callback at a time they choose, and end the call politely."
               if escalate else "")
        f, who = self.flow, self.flow.counterpart
        stages = "\n".join(f"{i}. {st.id} ({st.name}): {st.goal}" for i, st in enumerate(f.stages, 1))
        notes = self.flow_state["notes"]
        known = "; ".join(f"{k}={v}" for k, v in notes.items()) or "nothing yet"
        source = "IndiaMART's seller file" if f.mode == "seller" else "the call routing"
        meeting_tag = ("\nIf the seller has agreed a meeting day and time, append ##MEETING: <day and time>## to that reply."
                       if f.mode in MEETING_MODES else "")
        uc = sp.get("use_case")
        use_case = ""
        if uc and f.mode != "buyer":
            name = " / ".join(x for x in (uc.get("company"), uc.get("seller_name")) if x)
            todo = "; ".join(f"{c['key']} = {c['label']}" for c in uc["checklist"] if c["key"] not in notes)
            use_case = f"""
USE CASE: {uc['name']}. Goal: {uc['goal']}
APPROACH FOR THIS SELLER ({uc['archetype']['label']}, because {uc['archetype']['reason'].lower()}): {uc['archetype']['approach']}
CALL CONTEXT (from the seller file; already known, never ask for it again):{(' Seller: ' + name + '.') if name else ''}
{chr(10).join('- ' + c for c in uc['context']) or '- nothing specific'}
CHECKLIST still open: {todo or 'all done: confirm and close'}
Record each answer as ##NOTE: key=<the seller's answer>## with exactly these keys (e.g. listing_1=available, now 14 rupees per metre). When the seller's words already answer an item, record it and move to the next item; never ask them to confirm what they just said.
"""
        return f"""{f.role}
{use_case}

FACTS (the only facts you may state):
{facts}

GUARDRAILS (never break these):
{guard}

{who.upper()} (from {source}): {sp['summary']}
Personalisation hooks: {hooks}

CALL FLOW (a guide, not a script: move forward, skip what is already known, go back if they raise something):
{stages}
Current stage: {self.flow_state['stage']}. Already learned on this call: {known}.

COMMON QUESTIONS:
{chr(10).join('- ' + h for h in f.handlers)}

SOUND HUMAN (you are a person on a phone call, not a chatbot):
{chr(10).join('- ' + h for h in HUMAN_SPEECH + f.style)}
- On this call: {HUMAN_TOUCH.get(f.mode, HUMAN_TOUCH['seller'])}
- If the {who}'s last words seem cut off or they are still thinking, give a short encouraging acknowledgement and let them finish rather than changing topic.

DELIVERY FOR THIS {who.upper()} (updated live from how they talk):
- Language: {lang_name}. {MIX_TEXT.get(s.mix, '')} {script} Always answer in the {who}'s current language.
- Formality: {s.formality.replace('_', ' ')}. Warmth: {s.warmth}. Empathy: {s.empathy}. Vocabulary: {s.vocabulary}.
- Length: at most {LENGTH_WORDS[s.sentence_length]} words per reply, usually 1-2 short sentences; often less.

ACTIVE PERSONA: {PERSONAS[persona_id].name}. {s.focus}{esc}

OPENING PLAN: {' '.join(plan['opening']['steps'])}
OBJECTION PLAYBOOK:
{playbook or '- (see common questions)'}
ESCALATION RULES:
{escalation}

OUTPUT: only the words Payal speaks next, no stage directions, no markdown, no emoji. Never comment on your own speed or voice; just adapt. Write times and numbers as words. This is reply number {n_bot + 1}.
After the spoken words add ##STAGE: <stage id>## for the flow stage this reply works on, and, only if the {who} just told you something new, ##NOTE: key=value; key=value## (keys such as {', '.join(f.note_keys)}).
Only on your final goodbye (goal reached, {who} refused per the rules, or asked not to be called), append ##HANGUP##.{meeting_tag}"""

    def _messages(self, s: LiveSettings | None = None, persona_id: str | None = None,
                  escalate: bool | None = None, closing: str | None = None) -> list[dict]:
        s = s or self.settings
        persona_id = persona_id or self.persona_id
        escalate = self.escalate if escalate is None else escalate
        closing = closing or self.closing
        msgs = [{"role": "system", "content": self.system_prompt(s, persona_id, escalate)}]
        who = self.flow.counterpart
        for t in self.turns:
            # Past replies keep their stage tag so the model keeps writing it.
            tag = f"\n##STAGE: {t.meta['stage']}##" if t.role == "bot" and t.meta.get("stage") else ""
            if t.role == "bot" and t.meta.get("interrupted"):
                msgs.append({"role": "assistant", "content": t.text + " —" + tag})
                msgs.append({"role": "system", "content": f"(The {who} interrupted Payal here; she did not finish. "
                                                          "Respond to what they say next; don't repeat yourself.)"})
            elif t.role == "nudge":
                msgs.append({"role": "system", "content": f"(The {who} was silent for several seconds.)"})
            else:
                msgs.append({"role": "assistant" if t.role == "bot" else "user", "content": t.text + tag})
        if len(msgs) == 1:
            if self.flow.mode == "buyer":
                first = "(A buyer's call to a busy seller has reached you. Say your opening line.)"
            else:
                name = (self.spec.get("use_case") or {}).get("seller_name")
                first = ("(The seller has picked up the call. Open like a person: a greeting, that you are Payal from "
                         "IndiaMART, and check you are speaking with " + (f"{name}" if name else "the owner")
                         + ". Nothing else yet, at most 15 words: the reason comes after they answer.)")
            msgs.append({"role": "user", "content": first})
        # Restate the live settings last: models follow the latest instruction over the history's language.
        lang_name = sarvam.ALL_LANGUAGES.get(s.language, s.language)
        script = SCRIPT_TEXT.get(s.language, f"Use {lang_name} script.")
        if closing:
            meeting = f" Confirm the meeting once: {self.outcome['meeting']}." if self.outcome.get("meeting") else ""
            msgs.append({"role": "system", "content": (
                f"The {who} is wrapping up the call ({closing}). Reply in {lang_name} only. {script} End the call "
                "now, warmly, in at most 25 words: thank them for their time, say IndiaMART is there whenever they "
                f"need anything, and say goodbye.{meeting} No question, no new topic, no pitch. Then, on a new line, "
                "##HANGUP##")})
            return msgs
        ack = (f" You have already said '{self.ack}' out loud just now: carry straight on, without another "
               "acknowledgement (no ji / achha / theek hai at the start)." if self.ack else "")
        msgs.append({"role": "system", "content": (
            f"Reply now in {lang_name} only, whatever language earlier turns used. {script} "
            f"At most {LENGTH_WORDS[s.sentence_length]} words. Speak as {PERSONAS[persona_id].name}: {s.focus}"
            + (" Escalation: apologise, offer a callback, end the call." if escalate else "")
            + ack + self._avoid_repeats()
            + f" Then, on a new line, ##STAGE: <stage id>## and, if the {who} just shared a fact, "
              "##NOTE: key=value; ...##." + self._open_items_hint())})
        return msgs

    def _open_items_hint(self) -> str:
        uc = self.spec.get("use_case")
        if not uc or self.flow.mode == "buyer":
            return ""
        notes = self.flow_state["notes"]
        open_keys = [c["key"] for c in uc["checklist"] if c["key"] not in notes]
        if not open_keys:
            return ""
        return (f" Open checklist keys: {', '.join(open_keys)}. If the {self.flow.counterpart}'s last message answers "
                "any of them (even in passing, even approximately), you must write its NOTE now, and never ask "
                "that item again.")

    def _avoid_repeats(self) -> str:
        """Openers and phrases Payal has already used, so she does not sound like a loop."""
        bots = [t.text for t in self.turns if t.role == "bot"][-4:]
        if len(bots) < 2:
            return ""

        def words(text: str) -> list[str]:
            return [w.strip(",.।?!'\"") for w in text.lower().split() if w.strip(",.।?!'\"")]

        starts = []
        for b in bots[-3:]:
            first = " ".join(words(b)[:2])
            if first and first not in starts:
                starts.append(first)
        grams: Counter = Counter()
        for b in bots:
            w = words(b)
            grams.update({" ".join(w[i:i + 3]) for i in range(len(w) - 2)})
        repeated = [g for g, n in grams.most_common(8) if n >= 2][:5]
        out = " Vary your wording: do not start with " + " / ".join(f"'{x}'" for x in starts) + "."
        if repeated:
            out += " Do not reuse these phrases: " + "; ".join(f"'{g}'" for g in repeated) + "."
        return out

    # ---- voice ----------------------------------------------------------------------------
    def _tts_args(self, s: LiveSettings) -> tuple:
        lang = s.language if s.language in sarvam.TTS_LANGUAGES else "hi-IN"
        return lang, s.speaker, s.pace, s.expressiveness, 16000

    def speech_chunks(self, text: str, s: LiveSettings | None = None):
        """Yield (index, wav bytes, ms since start) in order; all sentences synthesise in parallel."""
        args = self._tts_args(s or self.settings)
        t0 = time.perf_counter()
        futs = [POOL.submit(sarvam.synthesize, p, *args) for p in split_speech(text)]
        for i, f in enumerate(futs):
            wav = f.result()[0]
            yield i, wav, int((time.perf_counter() - t0) * 1000)

    def speak(self, text: str, s: LiveSettings | None = None) -> tuple[bytes, int]:
        """Synthesize sentence-by-sentence in parallel and join, to cut latency."""
        t0 = time.perf_counter()
        wavs = [w for _, w, _ in self.speech_chunks(text, s)]
        return join_wavs(wavs), int((time.perf_counter() - t0) * 1000)

    def _generate(self, s: LiveSettings, persona_id: str, escalate: bool,
                  closing: str | None = None) -> tuple[str, int]:
        return sarvam.chat(self._messages(s, persona_id, escalate, closing), max_tokens=280, temperature=0.6)

    def _finish(self, content: str, ms: int) -> tuple[str, dict]:
        # The model sometimes drops the ## markers; never speak a tag out loud.
        content = mark_tags(content)
        meeting = re.search(r"##\s*MEETING:\s*([^<>#]+?)\s*##", content)
        if meeting and re.search(r"pending|unknown|tbd|day and time", meeting.group(1), re.I):
            meeting = None
        self._track_flow(content)
        text = re.sub(r"##.*?##|\[(END|MEETING:[^\]]*)\]", "", content, flags=re.S)
        text = re.sub(r"##.*$", "", text, flags=re.S)  # an unterminated tag at the end
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.।?!])", r"\1", text)          # a tag removed mid-sentence leaves " ," / " ."
        text = re.sub(r"[,]\s*([.।?!])", r"\1", text)
        text = drop_repeats_once(text, " ".join(t.text for t in self.turns if t.role == "bot"))
        text = trim_rambling(text, LENGTH_WORDS[self.settings.sentence_length])
        # A goodbye never asks a question, and the opening never ends the call.
        last_seller = next((t for t in reversed(self.turns) if t.role == "seller"), None)
        may_end = (bool(meeting) or "meeting" in self.outcome or self.escalate
                   or (self.flow.mode != "seller" and self.flow_state["stage"] == self.flow.stages[-1].id)
                   or (last_seller is not None and any(END_PATTERN.search(normalize(x or "")) for x in
                                                       (last_seller.text, last_seller.meta.get("romanized")))))
        end = "HANGUP" in content and may_end and not text.rstrip().endswith("?")
        if self.closing:
            # The caller is done: the call ends on this reply whatever the model wrote.
            if not text or text.rstrip().endswith("?"):
                text = GOODBYE_NOTE.get(self.settings.language, GOODBYE_NOTE["en-IN"])
            end = True
            self.outcome.setdefault("ended_by", f"{self.flow.counterpart} wrapped up the call: {self.closing}")
        if meeting:
            self.outcome["meeting"] = meeting.group(1).strip()
        notes = self.flow_state["notes"]
        if (self.flow.mode in MEETING_MODES and "meeting" not in self.outcome
                and notes.get("meeting_day") and notes.get("meeting_time")):
            mode = f", {notes['meeting_mode']}" if notes.get("meeting_mode") else ""
            self.outcome["meeting"] = f"{notes['meeting_day']} {notes['meeting_time']}{mode}"
        if end:
            self.ended = True
        return text, {"llm_ms": ms, "end": end, "stage": self.flow_state["stage"]}

    def _track_flow(self, content: str) -> None:
        stage = re.search(r"##\s*STAGE:\s*([a-z_]+)\s*##", content, re.I)
        ids = [st.id for st in self.flow.stages]
        if stage and stage.group(1).lower() in ids:
            sid = stage.group(1).lower()
            self.flow_state["stage"] = sid
            if sid not in self.flow_state["visited"]:
                self.flow_state["visited"].append(sid)
        for m in re.finditer(r"##\s*NOTE:\s*(.*?)\s*##", content, re.S):
            for pair in re.split(r"[;|\n]", m.group(1)):
                if "=" in pair:
                    k, v = (x.strip() for x in pair.split("=", 1))
                    k = re.sub(r"\W+", "_", k.lower()).strip("_")[:30]
                    if k and v and v.lower() not in ("none", "null", "unknown", "-"):
                        self.flow_state["notes"][k] = v[:80]

    def flow_status(self) -> dict:
        card = flow_card(self.spec)
        notes = self.flow_state["notes"]
        keys = [c["key"] for c in card["checklist"]]
        done = [k for k in keys if k in notes]
        progress = {"done": len(done), "total": len(keys)} if keys else None
        if progress:
            self.outcome["checklist"] = f"{len(done)}/{len(keys)}"
        return {**card, "stage": self.flow_state["stage"], "visited": self.flow_state["visited"],
                "notes": notes, "progress": progress}

    def _reply(self) -> tuple[str, dict]:
        return self._finish(*self._generate(self.settings, self.persona_id, self.escalate))

    # ---- turns ------------------------------------------------------------------------------
    def _with_audio(self, r: dict) -> dict:
        audio, tts_ms = self.speak(r["reply"])
        r["audio"] = audio
        r["timings"]["tts_ms"] = tts_ms
        if "reply_total_ms" in r["timings"]:
            r["timings"]["reply_total_ms"] += tts_ms
        return r

    def open(self, stream: bool = False) -> dict:
        self.warm_acks()
        text, meta = self._reply()
        self.turns.append(Turn("bot", text, meta))
        r = {"reply": text, "timings": dict(meta), "switches": [], "settings": self.settings.as_dict(),
             "ended": self.ended, "persona": self.persona_card(), "start_reason": self.start_reason,
             "timeline": self.timeline, "profile": self.profile.as_dict(), "flow": self.flow_status()}
        return r if stream else self._with_audio(r)

    def nudge(self, stream: bool = False) -> dict:
        """The caller has gone quiet: check they are still there, like a person would; close after MAX_NUDGES."""
        self.nudges += 1
        self.ack = None
        who = self.flow.counterpart
        self.turns.append(Turn("nudge", ""))
        if self.nudges > MAX_NUDGES:
            self.escalate = True
            instruction = (f"The {who} has not responded for a while. Say you cannot hear them and will call back "
                           "later, thank them, and end the call. Append ##HANGUP##.")
        else:
            instruction = (f"The {who} has been silent for several seconds. In one short line, check whether they can "
                           "hear you or are still there (e.g. 'Hello, kya aap line par hain?'), in their language. "
                           "Do not repeat your last message.")
        msgs = self._messages() + [{"role": "system", "content": instruction}]
        content, ms = sarvam.chat(msgs, max_tokens=80, temperature=0.4)
        if self.nudges > MAX_NUDGES and "HANGUP" not in content:
            content += " ##HANGUP##"
        text, meta = self._finish(content, ms)
        if self.nudges > MAX_NUDGES:
            self.ended = True
        self.turns.append(Turn("bot", text, {**meta, "nudge": self.nudges}))
        self.save()
        r = {"reply": text, "nudge": self.nudges, "timings": dict(meta), "switches": [],
             "settings": self.settings.as_dict(), "ended": self.ended, "persona": self.persona_card(),
             "timeline": self.timeline, "profile": self.profile.as_dict(), "flow": self.flow_status()}
        return r if stream else self._with_audio(r)

    def mark_interrupted(self) -> None:
        """The caller talked over Payal's last reply, so she did not finish it."""
        last = next((t for t in reversed(self.turns) if t.role == "bot"), None)
        if last is not None:
            last.meta["interrupted"] = True

    def persona_card(self) -> dict:
        p = PERSONAS[self.persona_id]
        return {"id": p.id, "name": p.name, "description": p.description, "colour": p.colour}

    def seller_audio(self, audio: bytes, mime: str = "audio/wav", stream: bool = False, on_ack=None) -> dict:
        t0 = time.perf_counter()
        f_native = POOL.submit(sarvam.transcribe, audio, "transcribe", "unknown", "turn.wav", mime)
        f_roman = POOL.submit(sarvam.transcribe, audio, "translit", "unknown", "turn.wav", mime)
        native = f_native.result()
        try:
            roman = f_roman.result()
        except sarvam.SarvamError:
            roman = native
        stt_ms = int((time.perf_counter() - t0) * 1000)
        audio_s = sarvam.audio_seconds(audio) if mime.startswith("audio/wav") else None
        return self._seller_turn(native.text, roman.text, native.language_code, native.language_probability,
                                 {"stt_ms": stt_ms}, audio_s, stream, on_ack=on_ack)

    def seller_text(self, text: str, language_code: str | None = None, stream: bool = False,
                    hold_if_unfinished: bool = True, on_ack=None) -> dict:
        """Typed fallback (or tests): treats the text as both native and romanized."""
        return self._seller_turn(text, text, language_code, 1.0 if language_code else None, {"stt_ms": 0},
                                 None, stream, hold_if_unfinished, on_ack=on_ack)

    def flush(self, stream: bool = False, on_ack=None) -> dict:
        """The caller stopped after an unfinished-sounding utterance: answer what we have."""
        if not self.pending:
            return {"transcript": "", "skipped": True, "switches": [], "settings": self.settings.as_dict()}
        p, self.pending = self.pending, None
        return self._seller_turn(p["native"], p["roman"], p["lang"], p["prob"], {"stt_ms": 0}, p["audio_s"], stream,
                                 hold_if_unfinished=False, merge=False, on_ack=on_ack)

    def _seller_turn(self, native: str, roman: str, lang: str | None, prob: float | None, timings: dict,
                     audio_s: float | None = None, stream: bool = False, hold_if_unfinished: bool = True,
                     merge: bool = True, on_ack=None) -> dict:
        if not native.strip():
            return {"transcript": "", "skipped": True, "switches": [], "settings": self.settings.as_dict()}
        self.nudges = 0
        if merge and self.pending:
            # The caller paused mid-thought and carried on: one utterance, not two turns.
            p, self.pending = self.pending, None
            native, roman = f"{p['native']} {native}", f"{p['roman']} {roman}"
            audio_s = (p["audio_s"] or 0) + (audio_s or 0) or None
            lang, prob = lang or p["lang"], prob if lang else p["prob"]
        unfinished, why = looks_incomplete(native)
        if not unfinished and roman != native:
            unfinished, why = looks_incomplete(roman)
        if hold_if_unfinished and unfinished and self.waits < MAX_WAITS:
            self.waits += 1
            self.pending = {"native": native, "roman": roman, "lang": lang, "prob": prob, "audio_s": audio_s}
            return {"transcript": native, "romanized": roman, "wait": True, "wait_reason": why, "switches": [],
                    "settings": self.settings.as_dict(), "timings": timings}
        self.waits = 0
        turn_no = sum(t.role == "seller" for t in self.turns) + 1
        done, done_phrase = wants_to_end([native, roman])
        if done and turn_no >= 2 and not self.closing:
            self.closing = f"they said '{done_phrase}'"
        ack = self.pick_ack(native) if on_ack and not self.closing else None
        self.ack = ack[0] if ack else None
        if ack:
            on_ack(*ack)
        self.turns.append(Turn("seller", native, {"romanized": roman, "language": lang, "language_probability": prob,
                                                  "audio_s": audio_s}))
        texts, n_words = [native, roman], len(native.split())
        t0 = time.perf_counter()

        # Draft the reply now with the current persona and a fast language guess; the analyst runs alongside.
        draft_profile = copy.deepcopy(self.profile)
        draft_lang = self.lang.peek(lang, prob, texts, n_words)
        if draft_lang in sarvam.TTS_LANGUAGES:
            draft_profile.language = draft_lang
            draft_profile.mix = mix_for(draft_lang, None, self.profile.mix, self.base.mix)
        draft_settings = self.compose(self.persona_id, draft_profile)
        draft_key = (self.persona_id, self.escalate, bool(self.closing),
                     tuple(getattr(draft_settings, k) for k in TEXT_KEYS))
        f_draft = POOL.submit(self._generate, draft_settings, self.persona_id, self.escalate, self.closing)

        convo = [(t.role, t.text) for t in self.turns if t.role in ("bot", "seller")]
        prev = self.verdicts[-1] if self.verdicts else None
        v = analyse(convo, roman, self.persona_id, prev, self.use_llm_signals, self.flow.counterpart)
        self.verdicts.append(v)
        if v.wants_to_end and turn_no >= 2 and not self.closing and "?" not in native:
            self.closing = "the analyst heard them signal they are done: " + (v.evidence or native)[:80]
        timings["analyst_ms"] = int((time.perf_counter() - t0) * 1000)

        # 1. Language.
        ld = self.lang.update(lang, prob, texts, n_words, hint=v.profile.get("language"))
        voiced, unvoiced = self.profile.language, self.unvoiced
        lang_reason = ""
        if ld:
            to = ld.detail["to"]
            name = sarvam.ALL_LANGUAGES.get(to, to)
            if to in sarvam.TTS_LANGUAGES:
                voiced, unvoiced = to, None
                lang_reason = f"Seller switched to {name}: reply in it."
            else:
                unvoiced = to
                lang_reason = (f"Seller spoke {name}; Saaras understands it but Sarvam has no voice for it, so reply "
                               "in simple Hindi and offer an executive who speaks it.")
        elif lang in sarvam.TTS_LANGUAGES and lang == self.profile.language:
            unvoiced = None
        self.lang.current = voiced
        mix = mix_for(voiced, v.profile.get("mix") if voiced == "hi-IN" else None, self.profile.mix, self.base.mix)

        # 2. Live seller profile (the persona generated from the conversation).
        prof_lang = copy.deepcopy(self.profile)
        prof_lang.language, prof_lang.mix = voiced, mix
        want, want_phrase = pace_request(texts)
        prof_changes = self.profile.observe(words=n_words, audio_s=audio_s, analyst=v.profile, pace_request=want,
                                            language=voiced, mix=mix, seller_turn=turn_no)
        prof_changes.pop("language", None)
        prof_changes.pop("mix", None)

        # 3. Persona.
        home = self.profile.home_persona
        home_reason = ("seller is impatient / wants it faster" if home == "quick_direct"
                       else "seller is relaxed again")
        if home == "normal" and self.rest_persona != "normal":
            home, home_reason = self.rest_persona, "seller is relaxed again; back to the persona this use case calls for"
        decision = self.director.decide(v, home=home, home_reason=home_reason)
        if decision.action == "escalate":
            self.escalate = True
        old_persona = self.persona_id

        # Attribute each setting change to its cause: language, then persona, then profile.
        s_before = self.settings
        self.unvoiced = unvoiced
        s_lang = self.compose(old_persona, prof_lang)
        s_persona = self.compose(decision.to_persona, prof_lang)
        if decision.switched:
            self.persona_id = decision.to_persona
            self.timeline.append({"seller_turn": turn_no, "persona": self.persona_id})
        self.settings = self.compose(self.persona_id, self.profile)

        def diff(a: LiveSettings, b: LiveSettings, keys=None) -> dict:
            da, db = a.as_dict(), b.as_dict()
            return {k: [da[k], db[k]] for k in (keys or da) if da[k] != db[k]}

        new_entries = []
        base_entry = {"call_id": self.id, "seller_turn": turn_no, "t_sec": round(time.time() - self.t0, 1),
                      "seller_said": native[:160]}
        lang_change = diff(s_before, s_lang, ("language", "mix", "vocabulary"))
        if ld and lang_change:
            new_entries.append({**base_entry, "type": "language", "signal": "language_switch",
                                "detector": ld.detector, "score": round(ld.score, 2), "evidence": ld.evidence[:120],
                                "from_persona": old_persona, "to_persona": old_persona,
                                "change": lang_change, "reason": lang_reason})
        if decision.switched:
            new_entries.append({**base_entry, "type": "persona", "signal": v.state, "detector": v.source,
                                "score": round(v.intensity, 2), "trend": v.trend, "evidence": (v.evidence or native)[:120],
                                "from_persona": decision.from_persona, "to_persona": decision.to_persona,
                                "from_name": PERSONAS[decision.from_persona].name,
                                "to_name": PERSONAS[decision.to_persona].name,
                                "change": diff(s_lang, s_persona, DELIVERY_KEYS), "reason": decision.reason,
                                "analyst_rationale": v.rationale})
        elif decision.action == "escalate":
            new_entries.append({**base_entry, "type": "escalation", "signal": v.state, "detector": v.source,
                                "score": round(v.intensity, 2), "trend": v.trend, "evidence": (v.evidence or native)[:120],
                                "from_persona": self.persona_id, "to_persona": self.persona_id,
                                "change": {"call": ["continue", "end with callback offer"]}, "reason": decision.reason})
        delivery_change = diff(s_persona, self.settings, ("pace", "sentence_length", "formality"))
        if prof_changes or delivery_change:
            why = []
            if want:
                why.append(f"asked for {want} speech ('{want_phrase}')")
            elif "pace_bias" in prof_changes:
                why.append(f"analyst: wants it {v.profile.get('pace')}")
            if "patience" in prof_changes:
                why.append(f"patience {prof_changes['patience'][0]} -> {prof_changes['patience'][1]}")
            if "formality" in prof_changes:
                why.append(f"talks {prof_changes['formality'][1].replace('_', ' ')}")
            if self.profile.speech_rate_wps and "pace" in delivery_change and not want:
                why.append(f"speaks at {self.profile.speech_rate_wps:.1f} words/s")
            new_entries.append({**base_entry, "type": "profile", "signal": "live_profile",
                                "detector": "phrase" if want else v.source,
                                "score": None, "evidence": (want_phrase or v.profile.get("persona") or native)[:120],
                                "from_persona": self.persona_id, "to_persona": self.persona_id,
                                "profile_change": prof_changes,
                                "change": delivery_change or {k: val for k, val in prof_changes.items()},
                                "reason": "Live seller profile: " + ("; ".join(why) or "updated") + "."})
        self.switch_log.extend(new_entries)

        # Keep the draft if the verdict did not change anything that shapes the text.
        final_key = (self.persona_id, self.escalate, bool(self.closing),
                     tuple(getattr(self.settings, k) for k in TEXT_KEYS))
        if final_key == draft_key:
            content, llm_ms = f_draft.result()
            timings["draft_reused"] = True
        else:
            content, llm_ms = self._generate(self.settings, self.persona_id, self.escalate, self.closing)
            timings["draft_reused"] = False
        text, meta = self._finish(content, llm_ms)
        ack, self.ack = self.ack, None
        if ack:
            text = LEADING_ACK.sub("", text, count=1).strip() or text
            text = text[:1].upper() + text[1:]
        timings["llm_ms"] = meta["llm_ms"]
        timings["reply_total_ms"] = int((time.perf_counter() - t0) * 1000) + timings["stt_ms"]
        self.turns.append(Turn("bot", f"{ack} {text}" if ack else text, {"settings": self.settings.as_dict()}))
        self.save()
        self.warm_acks()              # the voice may have changed (language / speaker)
        r = {"transcript": native, "romanized": roman, "language": lang, "language_probability": prob,
             "ack": ack,
             "analysis": v.as_dict(), "decision": decision.__dict__, "switches": new_entries, "reply": text,
             "settings": self.settings.as_dict(), "timings": timings, "ended": self.ended,
             "outcome": self.outcome, "persona": self.persona_card(), "timeline": self.timeline,
             "profile": self.profile.as_dict(), "flow": self.flow_status()}
        return r if stream else self._with_audio(r)

    def save(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        p = LOG_DIR / f"{self.id}.json"
        p.write_text(json.dumps({
            "call_id": self.id, "call_mode": self.flow.mode, "use_case": self.spec.get("use_case"),
            "seller_id": self.spec.get("seller_id"),
            "seller_type": self.spec["seller_type"],
            "start_persona": self.timeline[0]["persona"], "start_reason": self.start_reason,
            "persona_timeline": self.timeline,
            "start_settings": self.base.as_dict(), "final_settings": self.settings.as_dict(),
            "live_profile": self.profile.as_dict(), "profile_history": self.profile.history,
            "flow": self.flow_status(),
            "switch_log": self.switch_log, "outcome": self.outcome,
            "analyst_verdicts": [v.as_dict() for v in self.verdicts],
            "transcript": [{"role": t.role, "text": t.text} for t in self.turns],
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
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
