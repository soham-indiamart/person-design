import random

from vani_persona.matching.matcher import MatchResult
from vani_persona.personas import NEUTRAL_ID
from vani_persona.switching.state_machine import Action, PersonaSwitcher, SwitchConfig

A, B = "persona_a", "persona_b"


def res(i, a, b):
    return MatchResult("c", i, {A: a, B: b, NEUTRAL_ID: max(0.0, 1 - a - b)}, A, max(a, b), True, False, "")


def run(seq, cfg=None):
    sw = PersonaSwitcher(cfg or SwitchConfig())
    return sw, [sw.update(res(i, a, b), i + 1) for i, (a, b) in enumerate(seq)]


def test_waits_for_min_turns_and_confirmations():
    sw, ds = run([(0.9, 0.05), (0.9, 0.05), (0.9, 0.05)])
    assert [d.action for d in ds] == [Action.RETAIN, Action.RETAIN, Action.COMMIT]
    assert sw.state.active == A and sw.state.switches == 0  # first commit is free


def test_single_spike_does_not_switch():
    sw, ds = run([(0.9, 0.05)] * 3 + [(0.1, 0.85)] + [(0.9, 0.05)] * 3)
    assert sw.state.active == A and all(d.action != Action.SWITCH for d in ds)


def test_sustained_challenger_switches_once():
    sw, ds = run([(0.9, 0.05)] * 3 + [(0.1, 0.85)] * 4)
    assert sw.state.active == B
    assert sum(d.action == Action.SWITCH for d in ds) == 1


def test_margin_required():
    sw, _ = run([(0.9, 0.05)] * 3 + [(0.6, 0.7)] * 5)  # B leads, but only by 0.10 < 0.15
    assert sw.state.active == A


def test_no_oscillation_under_noise():
    rng = random.Random(0)
    seq = [(0.9, 0.05)] * 3 + [(p := rng.uniform(0.2, 0.75), 0.95 - p) for _ in range(40)]
    sw, ds = run(seq)
    changes = [d for d in ds if d.action in (Action.SWITCH, Action.REVERT_TO_NEUTRAL)]
    assert len(changes) <= SwitchConfig().max_switches_per_call


def test_cooldown_blocks_immediate_switch_back():
    cfg = SwitchConfig(min_dwell_seller_turns=1, switch_back_cooldown_turns=5, max_switches_per_call=5)
    sw, ds = run([(0.9, 0.05)] * 3 + [(0.1, 0.85)] * 2 + [(0.85, 0.1)] * 3, cfg)
    assert sw.state.active == B
    assert any("cooldown" in d.reason for d in ds)


def test_switch_budget_is_enforced():
    cfg = SwitchConfig(min_dwell_seller_turns=1, switch_back_cooldown_turns=0, max_switches_per_call=1)
    sw, _ = run([(0.9, 0.05)] * 3 + [(0.1, 0.85)] * 3 + [(0.85, 0.1)] * 5, cfg)
    assert sw.state.active == B and sw.state.switches == 1


def test_revert_to_neutral_when_current_persona_fades():
    sw, ds = run([(0.9, 0.05)] * 3 + [(0.2, 0.3)] * 3)
    assert ds[-1].action == Action.REVERT_TO_NEUTRAL and sw.state.active == NEUTRAL_ID


def test_matcher_outage_keeps_current_persona():
    sw, _ = run([(0.9, 0.05)] * 3)
    d = sw.update(MatchResult("c", 9, {}, A, 0, False, True, "scorer_error"), 4)
    assert d.action == Action.RETAIN and sw.state.active == A and "unavailable" in d.reason


def test_persona_specific_overrides_apply_while_active():
    # B beats A by 0.30: enough under the default margin (0.15), not under A's override (0.9).
    seq = [(0.9, 0.05)] * 3 + [(0.35, 0.65)] * 4
    base, _ = run(seq)
    assert base.state.active == B
    sw = PersonaSwitcher(SwitchConfig(), {A: {"switch_margin": 0.9}})
    for i, (a, b) in enumerate(seq):
        sw.update(res(i, a, b), i + 1)
    assert sw.state.active == A
    assert sw.cfg.switch_margin == 0.9 and sw.base_cfg.switch_margin == SwitchConfig().switch_margin


def test_override_keys_match_switch_config():
    import dataclasses

    from vani_persona.personas import SWITCH_OVERRIDE_KEYS

    fields = {f.name for f in dataclasses.fields(SwitchConfig)} - {"commit_threshold", "min_seller_turns_before_commit"}
    assert SWITCH_OVERRIDE_KEYS == fields
