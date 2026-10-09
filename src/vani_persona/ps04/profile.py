"""Live seller profile: the persona VANI builds of the seller *during* the call.

The seller file gives a starting guess. After every seller turn this profile is updated
from three sources:

- the Behaviour Analyst's read of the conversation (language, Hinglish mix, the pace the
  seller wants, patience, formality, a one-line description),
- explicit pace requests caught by phrase lexicons ("itna dheere kyu", "speak slowly"),
- how the seller actually talks: speaking rate (words per second of audio) and reply length.

The profile does two things: it picks the *home* persona the Director returns to when the
seller is calm (an impatient seller goes back to Quick & Direct, not Friendly Guide), and
it adjusts delivery on top of whichever persona is active (pace, sentence length,
formality). Explicit pace requests are sticky: switching persona never undoes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PACE_STEP = 0.12               # one explicit "faster" / "slower" request
PACE_BIAS_RANGE = (-0.3, 0.45)
RATE_NORMAL_WPS = 2.6          # conversational Hindi / Hinglish
RATE_BIAS_RANGE = (-0.1, 0.15)
PACE_RANGE = (0.6, 1.7)        # Bulbul accepts 0.5-2.0; beyond this speech gets hard to follow
TRAILING_SILENCE_S = 0.9       # the browser VAD keeps ~0.25 s before and ~0.65 s after each turn
LENGTH_ORDER = ["very_short", "short", "normal"]
PATIENCE = ("low", "medium", "high")


@dataclass
class LiveProfile:
    language: str
    mix: str
    formality: str
    patience: str = "medium"
    pace_bias: float = 0.0              # from explicit / analyst pace requests, sticky
    speech_rate_wps: float | None = None
    avg_words: float | None = None
    description: str = ""
    turns: int = 0
    calmer_reads: int = 0
    history: list[dict] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------------------
    @property
    def rate_bias(self) -> float:
        if self.speech_rate_wps is None:
            return 0.0
        lo, hi = RATE_BIAS_RANGE
        return round(min(hi, max(lo, (self.speech_rate_wps - RATE_NORMAL_WPS) * 0.08)), 3)

    @property
    def pace_wanted(self) -> str:
        total = self.pace_bias + self.rate_bias
        return "faster" if total >= 0.1 else "slower" if total <= -0.1 else "as is"

    @property
    def home_persona(self) -> str:
        """Persona to rest in while the seller is calm."""
        if self.patience == "low" or self.pace_bias >= PACE_STEP:
            return "quick_direct"
        return "normal"

    def traits(self) -> dict:
        return {"language": self.language, "mix": self.mix, "formality": self.formality,
                "patience": self.patience, "pace_wanted": self.pace_wanted}

    def label(self) -> str:
        bits = []
        if self.patience == "low":
            bits.append("Impatient")
        elif self.patience == "high":
            bits.append("Patient")
        if self.pace_wanted != "as is":
            bits.append(f"wants it {self.pace_wanted}")
        bits.append(self.formality.replace("_", " "))
        return " · ".join(bits)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("history")
        d.update(rate_bias=self.rate_bias, pace_wanted=self.pace_wanted, home_persona=self.home_persona,
                 label=self.label())
        return d

    # ---- update ----------------------------------------------------------------------
    def observe(self, *, words: int, audio_s: float | None, analyst: dict, pace_request: str | None,
                language: str, mix: str | None, seller_turn: int) -> dict:
        """Fold one seller turn in. Returns {trait: [old, new]} for the traits that changed."""
        before = self.traits() | {"pace_bias": round(self.pace_bias, 2)}
        self.turns += 1
        self.language = language
        if mix:
            self.mix = mix
        self.avg_words = words if self.avg_words is None else round(0.6 * self.avg_words + 0.4 * words, 1)
        if audio_s and words >= 3:
            voiced = max(0.6, audio_s - TRAILING_SILENCE_S)
            wps = words / voiced
            self.speech_rate_wps = round(wps if self.speech_rate_wps is None
                                         else 0.6 * self.speech_rate_wps + 0.4 * wps, 2)

        lo, hi = PACE_BIAS_RANGE
        if pace_request in ("faster", "slower"):
            step = PACE_STEP if pace_request == "faster" else -PACE_STEP
            self.pace_bias = round(min(hi, max(lo, self.pace_bias + step)), 2)
        elif analyst.get("pace") == "faster":
            # The analyst's read repeats every turn; only explicit requests stack beyond one step.
            self.pace_bias = max(self.pace_bias, PACE_STEP)
        elif analyst.get("pace") == "slower":
            self.pace_bias = min(self.pace_bias, -PACE_STEP)

        patience = analyst.get("patience")
        if patience in PATIENCE:
            # Impatience takes effect at once; leaving "low" needs two calmer reads in a row.
            if self.patience == "low" and patience != "low":
                self.calmer_reads += 1
                if self.calmer_reads >= 2:
                    self.patience, self.calmer_reads = patience, 0
            else:
                self.patience, self.calmer_reads = patience, 0
        elif self.avg_words is not None and self.turns >= 2 and self.avg_words <= 3:
            self.patience = "low"  # no analyst read: very short replies turn after turn
        if analyst.get("formality") in ("formal", "casual"):
            self.formality = "formal" if analyst["formality"] == "formal" else "friendly_casual"
        if analyst.get("persona"):
            self.description = str(analyst["persona"])[:120]

        after = self.traits() | {"pace_bias": round(self.pace_bias, 2)}
        changes = {k: [before[k], after[k]] for k in after if before[k] != after[k]}
        self.history.append({"seller_turn": seller_turn, "label": self.label(), "description": self.description,
                             "changes": changes})
        return changes


def profile_from_settings(language: str, mix: str, formality: str, seller_type: str) -> LiveProfile:
    return LiveProfile(language=language, mix=mix, formality=formality,
                       patience="low" if seller_type == "time_pressed" else "medium",
                       description="From the seller file; refined live from the conversation.")


def apply_profile(s, profile: LiveProfile) -> None:
    """Adjust a LiveSettings in place: pace, sentence length and formality from the profile."""
    lo, hi = PACE_RANGE
    s.pace = round(min(hi, max(lo, s.pace + profile.pace_bias + profile.rate_bias)), 2)
    if profile.patience == "low":
        cap = "very_short" if profile.pace_bias >= 2 * PACE_STEP else "short"
        if LENGTH_ORDER.index(s.sentence_length) > LENGTH_ORDER.index(cap):
            s.sentence_length = cap
    s.formality = profile.formality
    notes = []
    if profile.patience == "low":
        notes.append("The seller is impatient: no filler, no repetition, get to the point.")
    if profile.pace_wanted == "faster":
        notes.append("They want a brisker conversation.")
    elif profile.pace_wanted == "slower":
        notes.append("They want it slower and simpler: one idea per sentence.")
    if notes:
        s.focus = f"{s.focus} {' '.join(notes)}"
