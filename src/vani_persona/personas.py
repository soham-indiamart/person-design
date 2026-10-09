"""Persona loading, validation, safety lint and a versioned in-memory registry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from vani_persona import PERSONA_DIR, SCHEMA_DIR

NEUTRAL_ID = "neutral_default"

# Objection-handling moves a persona may use. Each one is a conversational move;
# none of them creates facts, offers or permissions. Any step that needs content
# ("..._from_policy", "answer_with_approved_fact") pulls it from the policy layer.
STEP_INSTRUCTIONS: dict[str, str] = {
    "acknowledge": "Briefly acknowledge what the seller said.",
    "acknowledge_concern": "Acknowledge the concern as reasonable without arguing.",
    "apologise_briefly": "Apologise in a few words for being unclear.",
    "answer_with_approved_fact": "Answer using only the approved answer for this topic from POLICY.",
    "answer_with_approved_fact_one_sentence": "Answer in one sentence using only the approved answer from POLICY.",
    "explain_what_is_included_from_policy": "Explain what is included, using only POLICY facts.",
    "state_identity_and_purpose": "Restate who is calling (VANI from IndiaMART) and why.",
    "state_identity_and_purpose_one_sentence": "In one sentence, restate who is calling and why.",
    "offer_verification_route_from_policy": "Offer the verification route listed in POLICY.",
    "rephrase_in_simpler_words": "Rephrase the last point in simpler, shorter words.",
    "give_one_example": "Give one concrete example relevant to the seller's products.",
    "confirm_understood": "Ask whether that answered their question.",
    "offer_details_on_whatsapp": "Offer to send the approved details on WhatsApp, if POLICY allows.",
    "offer_callback_slot_choice": "Offer two callback time options and let the seller choose.",
    "log_callback_and_close": "Confirm the callback time and close politely.",
    "ask_one_reason_once": "Ask once, gently, for the main reason. Do not ask again.",
    "respect_decision_and_close": "Accept the decision, thank the seller and close the call.",
    "return_to_objective": "Return to the next open step of the call objective.",
}

# Text that must never appear in a persona style: factual claims, offers and
# pressure tactics belong to (or are banned by) the policy layer.
FORBIDDEN_PATTERNS = [
    (re.compile(r"(₹|\brs\.?\s*\d|\d+\s*%)", re.I), "prices/percentages are facts and belong in POLICY"),
    (re.compile(r"\b(discount|cashback|free of cost|guarantee[d]?|refund)\b", re.I), "offers/guarantees belong in POLICY"),
    (re.compile(r"\b(limited time|last chance|only today|urgent|hurry|warn)\b", re.I), "pressure tactics are not allowed"),
    (re.compile(r"\b(ignore|override|bypass)\b.*\b(policy|disclosure|opt.?out|rule)", re.I), "personas cannot override policy"),
]


# Keys a persona may override in config/thresholds.yaml `switching` (must match SwitchConfig fields).
SWITCH_OVERRIDE_KEYS = {
    "switch_threshold", "switch_margin", "confirm_evaluations", "min_dwell_seller_turns",
    "switch_back_cooldown_turns", "max_switches_per_call", "revert_floor", "revert_evaluations",
}


class PersonaError(ValueError):
    pass


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads((SCHEMA_DIR / "persona.schema.json").read_text())


def _style_strings(style: dict) -> list[str]:
    out = [style.get("opening", ""), style.get("re_engagement", "")]
    out += [t["text"] for t in style.get("example_dialogue", []) if t.get("speaker") == "agent"]
    return out


def lint(persona: dict) -> list[str]:
    """Policy-safety checks beyond the JSON Schema. Returns a list of problems."""
    problems = []
    for key, steps in persona["style"]["objection_handling"].items():
        for s in steps:
            if s not in STEP_INSTRUCTIONS:
                problems.append(f"objection_handling.{key}: unknown step {s!r}")
    for key in persona.get("switching", {}).get("threshold_overrides", {}):
        if key not in SWITCH_OVERRIDE_KEYS:
            problems.append(f"switching.threshold_overrides: unknown key {key!r}")
    oh = persona["style"]["objection_handling"]
    if "default" not in oh:
        problems.append("objection_handling must define a 'default' sequence")
    # Disinterest must always be respected: at most one reason question, then close.
    ni = oh.get("not_interested")
    if ni is None:
        problems.append("objection_handling must define 'not_interested'")
    elif ni[-1] != "respect_decision_and_close" or ni.count("ask_one_reason_once") > 1:
        problems.append("objection_handling.not_interested must end with respect_decision_and_close")
    for text in _style_strings(persona["style"]):
        for pat, why in FORBIDDEN_PATTERNS:
            if pat.search(text):
                problems.append(f"style text {text[:50]!r}...: {why}")
    if persona["status"] in {"approved", "in_experiment"} and persona["persona_id"] != NEUTRAL_ID:
        ev = persona["definition"]["evidence"]
        if not ev.get("n_calls") or ev.get("segment_id") is None:
            problems.append("approved/in_experiment personas need segment_id and n_calls evidence")
    return problems


def validate(persona: dict) -> None:
    try:
        jsonschema.validate(persona, _schema())
    except jsonschema.ValidationError as e:
        raise PersonaError(f"{persona.get('persona_id', '?')}: schema: {e.message} at {list(e.absolute_path)}") from e
    problems = lint(persona)
    if problems:
        raise PersonaError(f"{persona['persona_id']}: " + "; ".join(problems))


def load_persona(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    validate(data)
    return data


@dataclass(frozen=True)
class PersonaRef:
    persona_id: str
    version: str


class PersonaRegistry:
    """Loads and validates every persona once, then serves them from memory by (id, version).

    In production this is backed by a config service. Loading at process start
    plus versioned keys means a call never sees a persona change mid-call: it
    pins the versions it started with.
    """

    def __init__(self, directory: Path = PERSONA_DIR, allowed_status: set[str] | None = None):
        self._by_key: dict[tuple[str, str], dict] = {}
        self._latest: dict[str, str] = {}
        for path in sorted(directory.glob("*.yaml")):
            p = load_persona(path)
            if allowed_status and p["status"] not in allowed_status and p["persona_id"] != NEUTRAL_ID:
                continue
            key = (p["persona_id"], p["version"])
            if key in self._by_key:
                raise PersonaError(f"duplicate persona version {key}")
            self._by_key[key] = p
            cur = self._latest.get(p["persona_id"])
            if cur is None or _semver(p["version"]) > _semver(cur):
                self._latest[p["persona_id"]] = p["version"]
        if NEUTRAL_ID not in self._latest:
            raise PersonaError("neutral_default persona is required")
        n = len(self._latest) - 1
        if n > 6:
            raise PersonaError(f"{n} non-neutral personas; the library allows at most 6")

    def get(self, persona_id: str, version: str | None = None) -> dict:
        version = version or self._latest[persona_id]
        return self._by_key[(persona_id, version)]

    def ref(self, persona_id: str) -> PersonaRef:
        return PersonaRef(persona_id, self._latest[persona_id])

    def ids(self, include_neutral: bool = False) -> list[str]:
        return [i for i in self._latest if include_neutral or i != NEUTRAL_ID]

    @property
    def neutral(self) -> dict:
        return self.get(NEUTRAL_ID)

    def snapshot(self) -> dict[str, str]:
        """Persona id -> version, pinned for the duration of one call."""
        return dict(self._latest)


def _semver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def persona_summary(p: dict[str, Any]) -> str:
    return f"{p['persona_id']}@{p['version']} ({p['status']})"
