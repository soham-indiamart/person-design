# Replay report (generated)

1745 held-out calls replayed turn by turn through `CallSession(mode="shadow")` with scorer `logistic` (`lr-202610090826`) and the thresholds in `config/thresholds.yaml`. Agreement is measured against full-call cluster labels, not human review.

## Switching behaviour

- Calls that never left neutral: 62.2%
- First commit at seller turn: {3: 0.343, 4: 0.252, 5: 0.152, 6: 0.076, 7: 0.062, 8: 0.039, 9: 0.03, 10: 0.017, 11: 0.008, 12: 0.009, 13: 0.005, 16: 0.002, 17: 0.003, 22: 0.002, 24: 0.002}
- Persona changes per call: {0: 0.622, 1: 0.262, 2: 0.101, 3: 0.015}
- Calls with > 1 change: 11.6% (Phase 3 target ≤ 5%, counting the first commit as a change)
- Calls with a switch between two personas: 2.3%; with a revert to neutral: 9.9%
- Calls that returned to an earlier persona (possible oscillation): 0.6%
- Max matcher latency within a call, p50 / p99: 1.62 / 3.38 ms (this machine)

## Final persona vs full-call segment

- Agreement: 82.5%

| true (segment persona) | calls | final = true | final distribution |
|---|---|---|---|
| neutral_default | 969 | 100.0% | neutral_default 100% |
| objection_raising | 483 | 56.5% | objection_raising 57%, neutral_default 38%, time_pressed 6% |
| time_pressed | 293 | 67.6% | time_pressed 68%, neutral_default 32% |
