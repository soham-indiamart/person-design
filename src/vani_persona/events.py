"""Call-event and outcome emission with idempotency, ordering and schema validation.

Events are append-only facts about what the system observed and did. Raw
transcripts are not included: transcripts live in the access-controlled
transcript store and are referenced by call_id.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from vani_persona import SCHEMA_DIR

EVENT_SCHEMA_VERSION = "1.0"
_NS = uuid.UUID("6f1c9a3e-4d0b-4f7a-9a51-2b9d6c0e8f11")

EVENT_TYPES = {
    "call_started",
    "match_evaluated",
    "persona_committed",
    "persona_switched",
    "persona_reverted",
    "style_override_applied",
    "opt_out_detected",
    "matcher_fallback",
    "agent_turn_composed",
    "call_ended",
}


@lru_cache(maxsize=4)
def _validator(name: str) -> jsonschema.Draft202012Validator:
    # Compiled once: jsonschema.validate() re-checks the schema on every call (~3 ms).
    schema = json.loads((SCHEMA_DIR / name).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLog:
    """Per-call ordered event emitter. ``sink`` is a JSONL path (a stream/queue in production).

    ``event_id`` is derived from (call_id, seq, event_type), so a redelivered event
    keeps its id and consumers deduplicate on it. ``seq`` gives a total order within a call.
    """

    def __init__(self, call_id: str, session_id: str, experiment: dict | None, sink: Path | None = None):
        self.call_id = call_id
        self.session_id = session_id
        self.experiment = experiment or {"experiment_id": None, "arm": None}
        self.sink = sink
        self.seq = 0
        self.events: list[dict] = []

    def emit(self, event_type: str, persona: dict | None, payload: dict[str, Any], turn_index: int | None = None) -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type}")
        self.seq += 1
        event_id = str(uuid.uuid5(_NS, f"{self.call_id}:{self.seq}:{event_type}"))
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": event_id,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "event_type": event_type,
            "ts": _now(),
            "turn_index": turn_index,
            "experiment": self.experiment,
            "persona": persona,
            "payload": payload,
        }
        _validator("call_event.schema.json").validate(event)
        self.events.append(event)
        if self.sink:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


def outcome_record(
    call_id: str, outcome_type: str, value: Any, source: str, observed_at: str | None = None, details: dict | None = None
) -> dict:
    """An outcome can arrive minutes (disposition) or weeks (conversion) after the call.

    ``outcome_id`` is deterministic, so resubmitting the same outcome is idempotent.
    A corrected value is sent as a new record with ``supersedes``.
    """
    rec = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "outcome_id": str(uuid.uuid5(_NS, f"{call_id}:{outcome_type}:{source}")),
        "call_id": call_id,
        "outcome_type": outcome_type,
        "value": value,
        "source": source,
        "observed_at": observed_at or _now(),
        "supersedes": None,
        "details": details or {},
    }
    _validator("outcome.schema.json").validate(rec)
    return rec
