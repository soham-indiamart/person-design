import copy
import shutil

import pytest
import yaml

from vani_persona import PERSONA_DIR
from vani_persona.personas import NEUTRAL_ID, PersonaError, PersonaRegistry, lint, load_persona, validate


@pytest.mark.parametrize("path", sorted(PERSONA_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_persona_file_validates(path):
    load_persona(path)


def test_only_approved_personas_may_be_marked_validated(registry):
    for pid in registry.ids(include_neutral=True):
        p = registry.get(pid)
        if p["status"] != "approved":
            assert p["evaluation"]["validated"] is False


def test_library_size_within_brief(registry):
    # 3-6 evidence-backed personas, counting neutral_default (it carries segment 0's evidence)
    assert 3 <= len(registry.ids(include_neutral=True)) <= 6
    for pid in registry.ids(include_neutral=True):
        ev = registry.get(pid)["definition"]["evidence"]
        assert ev["n_calls"] and ev["segment_id"] is not None


@pytest.mark.parametrize("path", sorted((PERSONA_DIR / "retired").glob("*.yaml")), ids=lambda p: p.name)
def test_retired_personas_still_validate_and_are_not_loaded(path, registry):
    p = load_persona(path)
    assert p["status"] == "retired" and p["persona_id"] not in registry.ids()


def _base():
    return load_persona(PERSONA_DIR / "time_pressed.yaml")


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda p: p["style"].__setitem__("opening", "Offer a 20% discount today"), "POLICY"),
        (lambda p: p["style"].__setitem__("re_engagement", "Tell them it is a last chance"), "pressure"),
        (lambda p: p["style"]["objection_handling"].__setitem__("price", ["promise_free_upgrade"]), "unknown step"),
        (lambda p: p["style"]["objection_handling"].pop("default"), "default"),
        (lambda p: p["style"]["objection_handling"].__setitem__("not_interested", ["acknowledge", "return_to_objective"]),
         "respect_decision_and_close"),
        (lambda p: p["style"]["objection_handling"].pop("not_interested"), "not_interested"),
        (lambda p: (p.__setitem__("status", "approved"), p["definition"]["evidence"].__setitem__("n_calls", None)),
         "evidence"),
    ],
)
def test_lint_rejects_unsafe_personas(mutate, expect):
    p = copy.deepcopy(_base())
    mutate(p)
    problems = lint(p)
    assert any(expect in x for x in problems), problems


def test_schema_rejects_out_of_range_tts():
    p = copy.deepcopy(_base())
    p["style"]["pace"]["tts_rate"] = 1.6
    with pytest.raises(PersonaError):
        validate(p)


def test_registry_requires_neutral(tmp_path):
    shutil.copy(PERSONA_DIR / "objection_raising.yaml", tmp_path)
    with pytest.raises(PersonaError, match="neutral_default"):
        PersonaRegistry(tmp_path)


def test_registry_caps_library_at_six(tmp_path):
    shutil.copy(PERSONA_DIR / "neutral_default.yaml", tmp_path)
    base = yaml.safe_load((PERSONA_DIR / "objection_raising.yaml").read_text())
    for i in range(7):
        p = copy.deepcopy(base)
        p["persona_id"] = f"extra_{i}"
        (tmp_path / f"p{i}.yaml").write_text(yaml.safe_dump(p))
    with pytest.raises(PersonaError, match="at most 6"):
        PersonaRegistry(tmp_path)


def test_registry_status_filter_keeps_neutral():
    r = PersonaRegistry(allowed_status={"approved"})
    assert r.ids() == [] and r.neutral["persona_id"] == NEUTRAL_ID
