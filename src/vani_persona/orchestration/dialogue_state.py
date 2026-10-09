"""Conversation state and business policy. Both are owned by the orchestrator and independent of personas.

A persona switch replaces only the STYLE layer. Nothing in this module is read
from, or written by, persona configuration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from vani_persona import CONFIG_DIR


class Stage(str, Enum):
    OPENING = "opening"
    DISCOVERY = "discovery"
    PROPOSAL = "proposal"
    OBJECTION = "objection"
    CLOSING = "closing"
    ENDED = "ended"


class ConsentState(BaseModel):
    ai_disclosure_given: bool = False
    recording_disclosure_given: bool = False
    opted_out: bool = False


class DialogueState(BaseModel):
    call_id: str
    objective_id: str
    stage: Stage = Stage.OPENING
    language: str = "hi"
    agent_turns: int = 0
    collected_facts: dict[str, str] = Field(default_factory=dict)
    answered_questions: list[str] = Field(default_factory=list)
    open_objections: list[str] = Field(default_factory=list)
    resolved_objections: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    consent: ConsentState = Field(default_factory=ConsentState)


class ObjectivePolicy(BaseModel):
    """Business rules for one call objective. Versioned and approved separately from personas."""

    objective_id: str
    version: str
    description: str
    steps: list[str]
    mandatory_disclosures: list[str]
    approved_answers: dict[str, str]
    verification_route: str
    allowed_actions: list[str]
    forbidden: list[str]
    supported_languages: list[str]

    @classmethod
    def load(cls, objective_id: str, directory: Path | None = None) -> "ObjectivePolicy":
        path = (directory or CONFIG_DIR / "objectives") / f"{objective_id}.yaml"
        return cls(**yaml.safe_load(path.read_text()))

    def next_step(self, state: DialogueState) -> str | None:
        return next((s for s in self.steps if s not in state.completed_steps), None)
