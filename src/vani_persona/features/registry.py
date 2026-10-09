"""Feature metadata: what each behavioural feature means, where it comes from, and how far to trust it.

``role`` decides how a feature is used:
  segment   - defines personas (behavioural, dense, transcript-derivable, live-available)
  detail    - behavioural but sparse or redundant: describes segments and drives objection
              strategy, but is NOT clustered on. Zero-inflated lexicon rates destabilize clustering:
              on the PS07 export no k passed the gates when they were included
  language  - drives the separate language/locale axis; never used to cluster personas
  context   - reported alongside segments (e.g. duration) but never used to define them
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    definition: str
    extraction: str
    sources: tuple[str, ...]
    availability: str  # "historical" | "live" | "both"
    reliability: str  # "high" | "medium" | "low"
    failure_modes: tuple[str, ...]
    role: str = "segment"
    requires_timestamps: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


FEATURES: dict[str, FeatureSpec] = {
    f.name: f
    for f in [
        FeatureSpec(
            "n_seller_turns", "turn_taking", "Number of seller turns observed so far.",
            "Count of turns with speaker=seller.", ("transcript",), "both", "high",
            ("ASR diarization errors merge or split turns",), role="context",
        ),
        FeatureSpec(
            "seller_mean_tokens", "communication_preference",
            "Mean whitespace tokens per seller turn.", "Token count per seller turn, averaged.",
            ("transcript",), "both", "medium",
            ("Devanagari vs romanized tokenization differ", "ASR drops words in noise"),
            role="detail",
        ),
        FeatureSpec(
            "short_reply_ratio", "turn_taking", "Share of seller turns with at most 3 tokens.",
            "Count(tokens<=3)/seller turns.", ("transcript",), "both", "medium",
            ("Backchannels ('haan', 'ji') are legitimately short",
             "Short answers to yes/no questions inflate this"),
        ),
        FeatureSpec(
            "long_reply_ratio", "engagement", "Share of seller turns with at least 15 tokens.",
            "Count(tokens>=15)/seller turns.", ("transcript",), "both", "medium",
            ("ASR merging two turns into one",),
            role="detail",
        ),
        FeatureSpec(
            "question_rate", "engagement",
            "Share of seller turns that ask something ('?' or a question word).",
            "Lexicon + punctuation per seller turn.", ("transcript",), "both", "low",
            ("ASR rarely emits '?'", "'kya' is also a filler / yes-no marker"),
        ),
        FeatureSpec(
            "clarification_rate", "communication_preference",
            "Share of seller turns asking to repeat or clarify.", "CLARIFICATION lexicon.",
            ("transcript",), "both", "medium",
            ("Often caused by line noise or TTS clarity, not by the seller's preference",),
            # A direct request, handled by the explicit-override layer (repeated_clarification).
            # On PS07 the genuine-clarification group was stable but only 3.5% of calls.
            role="detail",
        ),
        FeatureSpec(
            "brevity_request_rate", "communication_preference",
            "Share of seller turns explicitly asking VANI to be brief.", "BREVITY_REQUEST lexicon.",
            ("transcript",), "both", "high", ("Rare event: sparse signal",),
            role="detail",
        ),
        FeatureSpec(
            "time_pressure_rate", "engagement",
            "Share of seller turns stating they are busy / want a later call.",
            "TIME_PRESSURE lexicon.", ("transcript",), "both", "medium",
            ("'later' also appears in unrelated sentences",),
            role="detail",
        ),
        FeatureSpec(
            "objection_price_rate", "objection", "Share of seller turns raising cost/fees.",
            "OBJECTION_PRICE lexicon.", ("transcript",), "both", "medium",
            ("A neutral price question is not always an objection",),
            role="detail",
        ),
        FeatureSpec(
            "objection_trust_rate", "objection",
            "Share of seller turns questioning legitimacy (spam/fraud/who is calling).",
            "OBJECTION_TRUST lexicon.", ("transcript",), "both", "medium",
            ("'Who is this' can be simple identification",),
            role="detail",
        ),
        FeatureSpec(
            "objection_not_interested_rate", "objection",
            "Share of seller turns stating disinterest.", "OBJECTION_NOT_INTERESTED lexicon.",
            ("transcript",), "both", "high", (),
            role="detail",
        ),
        FeatureSpec(
            "objection_already_have_rate", "objection",
            "Share of seller turns saying it is already done/available.",
            "OBJECTION_ALREADY_HAVE lexicon.", ("transcript",), "both", "low",
            ("'already' is common in neutral speech",),
            role="detail",
        ),
        FeatureSpec(
            "info_request_rate", "objection",
            "Share of seller turns asking for details to be sent (WhatsApp/email).",
            "INFO_REQUEST lexicon.", ("transcript",), "both", "medium",
            ("Can be a polite deferral rather than real interest",),
            role="detail",
        ),
        FeatureSpec(
            "affirmation_rate", "conversation_progression",
            "Share of seller turns containing an affirmation (haan/ji/ok/theek hai).",
            "AFFIRMATION lexicon.", ("transcript",), "both", "medium",
            ("Backchannel 'ji' is politeness, not agreement",),
        ),
        FeatureSpec(
            "engagement_slope", "conversation_progression",
            "Linear slope of seller tokens per turn over seller turn order (tokens/turn).",
            "OLS slope over seller turns; needs >= 3 seller turns.", ("transcript",), "both", "low",
            ("Undefined for short calls", "Driven by the agent's question type"),
            role="detail",
        ),
        FeatureSpec(
            "response_latency_median_ms", "speaking_pace",
            "Median gap between the end of an agent turn and the start of the next seller turn.",
            "Timestamps on consecutive agent->seller turns; negative gaps clipped to 0.",
            ("transcript_timestamps",), "both", "medium",
            ("Includes ASR endpointing delay", "Missing when export has no timestamps"),
            requires_timestamps=True,
            role="detail",
        ),
        FeatureSpec(
            "overlap_rate", "turn_taking",
            "Share of seller turns that start before the preceding agent turn ended "
            "(or agent turns flagged interrupted when timestamps are absent).",
            "Timestamp overlap, else the 'interrupted' flag.",
            ("transcript_timestamps", "telephony_events"), "both", "medium",
            ("Echo / crosstalk false positives", "Unavailable without timestamps or barge-in flags"),
            requires_timestamps=True,
            role="detail",
        ),
        FeatureSpec(
            "log_mean_tokens", "communication_preference",
            "log(1 + mean tokens per seller turn): verbosity on a scale robust to a few long monologues.",
            "Derived from seller_mean_tokens.", ("transcript",), "both", "medium",
            ("Devanagari vs romanized tokenization differ",),
        ),
        FeatureSpec(
            "resistance_rate", "objection",
            "Share of seller turns raising ANY objection (price, trust, not interested, already have).",
            "Union of the objection lexicons.", ("transcript",), "both", "medium",
            ("Inherits each lexicon's false positives",),
        ),
        FeatureSpec(
            "rush_rate", "engagement",
            "Share of seller turns signalling time pressure or asking VANI to be brief.",
            "Union of TIME_PRESSURE and BREVITY_REQUEST lexicons.", ("transcript",), "both", "medium",
            ("'baad mein' in unrelated sentences",),
        ),
        FeatureSpec(
            "devanagari_ratio", "language", "Share of seller letters in Devanagari script.",
            "Unicode block count.", ("transcript",), "both", "high",
            ("Reflects the ASR model's output script, not necessarily the spoken language",),
            role="language",
        ),
        FeatureSpec(
            "hindi_function_ratio", "language",
            "Hindi function words / (Hindi + English function words) in seller speech.",
            "Function-word lexicons; Devanagari tokens count as Hindi.", ("transcript",), "both",
            "medium", ("Low token counts make the ratio noisy",), role="language",
        ),
        FeatureSpec(
            "code_switch_rate", "language",
            "Share of seller turns containing both Hindi and English function words.",
            "Function-word lexicons per turn.", ("transcript",), "both", "low",
            ("Only detects function-word mixing; misses noun-only English insertions",),
            role="language",
        ),
        FeatureSpec(
            "duration_s", "outcome_context", "Call duration in seconds.",
            "Max end - min start.", ("transcript_timestamps",), "historical", "high",
            ("Short call != impatient seller: also caused by drops, wrong numbers, quick success",),
            role="context", requires_timestamps=True,
        ),
    ]
}


def segment_features() -> list[str]:
    return [n for n, f in FEATURES.items() if f.role == "segment"]


def language_features() -> list[str]:
    return [n for n, f in FEATURES.items() if f.role == "language"]
