import numpy as np
import pandas as pd
import pytest

from vani_persona.experiment.analysis import Effect, Guardrail, check_guardrails, cluster_bootstrap_effect, srm_pvalue
from vani_persona.experiment.assignment import assign_arm, is_exploration_call
from vani_persona.experiment.power import design_effect, n_per_arm, plan


def test_assignment_deterministic_and_balanced():
    alloc = {"control": 0.5, "treatment": 0.5}
    arms = [assign_arm(f"s{i}", "exp1", alloc) for i in range(20000)]
    assert arms == [assign_arm(f"s{i}", "exp1", alloc) for i in range(20000)]
    share = arms.count("treatment") / len(arms)
    assert 0.48 < share < 0.52


def test_assignment_independent_across_experiments():
    alloc = {"control": 0.5, "treatment": 0.5}
    a = [assign_arm(f"s{i}", "exp1", alloc) for i in range(5000)]
    b = [assign_arm(f"s{i}", "exp2", alloc) for i in range(5000)]
    agree = np.mean([x == y for x, y in zip(a, b)])
    assert 0.45 < agree < 0.55


def test_allocation_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_arm("s", "e", {"control": 0.5, "treatment": 0.4})


def test_exploration_rate():
    rate = np.mean([is_exploration_call(f"c{i}", "e", 0.05) for i in range(20000)])
    assert 0.04 < rate < 0.06


def test_power_matches_textbook_value():
    # 50% -> 60%, alpha 0.05 two-sided, power 0.8: ~385 per arm (Fleiss, no continuity correction)
    assert 383 <= n_per_arm(0.5, 0.1) <= 389


def test_design_effect_inflates_sample():
    assert design_effect(1, 0.3) == 1
    p = plan(0.2, 0.02, 0.05, 0.8, calls_per_seller=3, icc=0.1, daily_calls=1000)
    assert p["calls_per_arm"] == pytest.approx(p["calls_per_arm_iid"] * 1.2, rel=0.01)
    assert p["weeks_rounded"] >= 1


def test_srm_detects_broken_split():
    assert srm_pvalue(5000, 5000) > 0.5
    assert srm_pvalue(5000, 5600) < 0.001


def _df(p_c, p_t, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        arm = "treatment" if i % 2 else "control"
        rows.append({"seller_key": f"s{i // 2}{arm}", "arm": arm, "y": rng.random() < (p_t if arm == "treatment" else p_c)})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_finds_real_effect_and_not_null():
    eff = cluster_bootstrap_effect(_df(0.2, 0.3), "y", n_boot=500)
    assert eff.significant() and eff.diff_abs > 0
    null = cluster_bootstrap_effect(_df(0.2, 0.2, seed=1), "y", n_boot=500)
    assert null.ci95[0] < 0 < null.ci95[1]


def test_guardrails():
    eff = {"opt_out_rate": Effect("opt_out_rate", 0.02, 0.05, 0.03, (0.02, 0.04), 1.5, 2000, 2000)}
    breaches = check_guardrails(eff, [Guardrail("opt_out_rate", "lower_is_better", 0.005), Guardrail("latency", "lower_is_better", 1)])
    assert any("opt_out_rate" in b for b in breaches) and any("not measured" in b for b in breaches)
