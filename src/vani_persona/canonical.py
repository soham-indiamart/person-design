"""Canonical call record.

Every raw export is mapped into these models by ``ingest.adapter``. Observed data
(what was said, when, and what disposition was logged) is kept apart from inferred
labels (segments, persona predictions), which never live on ``CallRecord``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

CANONICAL_SCHEMA_VERSION = "1.0"


class Speaker(str, Enum):
    AGENT = "agent"
    SELLER = "seller"
    UNKNOWN = "unknown"


class DispositionClass(str, Enum):
    """Normalized disposition. The raw disposition string is always kept as well."""

    OBJECTIVE_COMPLETED = "objective_completed"
    CALLBACK_REQUESTED = "callback_requested"
    NOT_INTERESTED = "not_interested"
    OPT_OUT = "opt_out"
    SELLER_HUNG_UP = "seller_hung_up"
    NO_RESPONSE = "no_response"
    WRONG_PARTY = "wrong_party"  # wrong number / not the decision maker: no seller behaviour to learn from
    TECHNICAL_FAILURE = "technical_failure"
    OTHER = "other"


class Turn(BaseModel):
    idx: int = Field(ge=0, description="0-based position in the call")
    speaker: Speaker
    text: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    asr_confidence: float | None = Field(default=None, ge=0, le=1)
    # Observed only if the platform logged barge-in / truncated TTS for this turn.
    interrupted: bool | None = None


class CallContext(BaseModel):
    """Metadata that may confound outcomes. Never used to define personas."""

    campaign_id: str | None = None
    call_objective: str | None = None
    lead_type: str | None = Field(default=None, description="Lead bucket known before dialing (lead quality confounder)")
    seller_category: str | None = None
    language_hint: str | None = None
    started_at: datetime | None = None
    agent_version: str | None = None
    script_version: str | None = None
    region: str | None = None


class Outcome(BaseModel):
    disposition_raw: str | None = None
    disposition_class: DispositionClass = DispositionClass.OTHER
    objective_achieved: bool | None = None
    downstream_conversion: bool | None = None
    termination_reason: str | None = None
    observed_at: datetime | None = None


class DataQuality(BaseModel):
    has_timestamps: bool = False
    has_asr_confidence: bool = False
    mean_asr_confidence: float | None = None
    redactions_applied: int = 0
    unknown_speaker_turns: int = 0


class CallRecord(BaseModel):
    schema_version: str = CANONICAL_SCHEMA_VERSION
    call_id: str
    seller_key: str | None = Field(
        default=None, description="Pseudonymized seller identifier (salted hash), never the raw ID"
    )
    context: CallContext = Field(default_factory=CallContext)
    turns: list[Turn] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    quality: DataQuality = Field(default_factory=DataQuality)

    def seller_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == Speaker.SELLER]

    def duration_s(self) -> float | None:
        ends = [t.end_ms for t in self.turns if t.end_ms is not None]
        starts = [t.start_ms for t in self.turns if t.start_ms is not None]
        if not ends or not starts:
            return None
        return (max(ends) - min(starts)) / 1000.0
