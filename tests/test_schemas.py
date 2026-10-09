import json

import jsonschema
import pytest
from referencing import Registry, Resource

from vani_persona import SCHEMA_DIR
from vani_persona.schemagen import GENERATED, render

EXAMPLES = {
    "match_request.json": "match_request.schema.json",
    "match_response.json": "match_response.schema.json",
    "switch_decision.json": "switch_decision.schema.json",
    "call_event.persona_switched.json": "call_event.schema.json",
    "call_event.match_evaluated.json": "call_event.schema.json",
    "outcome.json": "outcome.schema.json",
}


def _registry():
    resources = []
    for p in SCHEMA_DIR.glob("*.schema.json"):
        s = json.loads(p.read_text())
        resources.append((p.name, Resource.from_contents(s)))
        resources.append((s["$id"], Resource.from_contents(s)))
    return Registry().with_resources(resources)


@pytest.mark.parametrize("name", sorted(GENERATED))
def test_generated_schemas_are_current(name):
    assert (SCHEMA_DIR / name).read_text() == render(GENERATED[name]), "run: python -m vani_persona.schemagen"


@pytest.mark.parametrize("path", sorted(SCHEMA_DIR.glob("*.schema.json")), ids=lambda p: p.name)
def test_schemas_are_valid_draft_2020_12(path):
    jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


@pytest.mark.parametrize("example,schema", sorted(EXAMPLES.items()))
def test_examples_validate(example, schema):
    s = json.loads((SCHEMA_DIR / schema).read_text())
    doc = json.loads((SCHEMA_DIR / "examples" / example).read_text())
    jsonschema.Draft202012Validator(s, registry=_registry()).validate(doc)


def test_match_request_rejects_unknown_feature():
    s = json.loads((SCHEMA_DIR / "match_request.schema.json").read_text())
    doc = json.loads((SCHEMA_DIR / "examples" / "match_request.json").read_text())
    doc["features"]["seller_gender"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(s, registry=_registry()).validate(doc)
