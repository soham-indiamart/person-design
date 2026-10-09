"""Regenerate the schemas derived from code: features (from the registry) and canonical_call (from pydantic).

    python -m vani_persona.schemagen
``tests/test_schemas.py`` fails if the committed files are out of date.
"""

from __future__ import annotations

import json

from vani_persona import SCHEMA_DIR
from vani_persona.canonical import CallRecord
from vani_persona.features.registry import FEATURES


def features_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://vani.internal/schemas/features.schema.json",
        "title": "Behavioural feature vector (full call or live prefix)",
        "description": "null = not observable (e.g. no timestamps), which is different from 0. "
        "Generated from vani_persona.features.registry.",
        "type": "object",
        "additionalProperties": False,
        "required": ["n_seller_turns"],
        "properties": {
            name: {
                "type": ["number", "null"],
                "description": f"[{spec.role}/{spec.category}, reliability={spec.reliability}] {spec.definition}",
            }
            for name, spec in FEATURES.items()
        },
    }


def canonical_schema() -> dict:
    s = CallRecord.model_json_schema()
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["$id"] = "https://vani.internal/schemas/canonical_call.schema.json"
    return s


GENERATED = {"features.schema.json": features_schema, "canonical_call.schema.json": canonical_schema}


def render(fn) -> str:
    return json.dumps(fn(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    for name, fn in GENERATED.items():
        (SCHEMA_DIR / name).write_text(render(fn))
        print(f"wrote schemas/{name}")


if __name__ == "__main__":
    main()
