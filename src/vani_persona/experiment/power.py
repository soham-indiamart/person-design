"""Sample size for a two-arm test on a binary metric.

    python -m vani_persona.experiment.power --baseline 0.18 --mde 0.02 --calls-per-seller 1.6 --icc 0.05

The baseline must come from the measured baseline period (Phase 1), not from a guess.
"""

from __future__ import annotations

import argparse
import math

from scipy.stats import norm


def n_per_arm(p_control: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Calls per arm to detect an absolute lift of ``mde_abs`` (two-sided, unpooled-variance alternative)."""
    p1, p2 = p_control, p_control + mde_abs
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("rates must be in (0, 1)")
    pbar = (p1 + p2) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / mde_abs**2)


def design_effect(calls_per_seller: float, icc: float) -> float:
    """Randomizing by seller while measuring per call: variance inflates by 1 + (m - 1) * ICC."""
    return 1 + max(calls_per_seller - 1, 0) * icc


def plan(p_control: float, mde_abs: float, alpha: float, power: float, calls_per_seller: float, icc: float, daily_calls: float | None) -> dict:
    base = n_per_arm(p_control, mde_abs, alpha, power)
    de = design_effect(calls_per_seller, icc)
    n = math.ceil(base * de)
    out = {"calls_per_arm_iid": base, "design_effect": round(de, 3), "calls_per_arm": n, "total_calls": 2 * n}
    if daily_calls:
        # Round up to whole weeks so weekday/weekend mix is balanced.
        days = math.ceil(2 * n / daily_calls)
        out["days"] = days
        out["weeks_rounded"] = math.ceil(days / 7)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=float, required=True, help="control rate from the baseline period")
    p.add_argument("--mde", type=float, required=True, help="minimum detectable ABSOLUTE lift, e.g. 0.02")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--calls-per-seller", type=float, default=1.0)
    p.add_argument("--icc", type=float, default=0.0, help="intra-seller correlation, estimated from baseline data")
    p.add_argument("--daily-calls", type=float, default=None, help="eligible calls per day across both arms")
    a = p.parse_args()
    for k, v in plan(a.baseline, a.mde, a.alpha, a.power, a.calls_per_seller, a.icc, a.daily_calls).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
