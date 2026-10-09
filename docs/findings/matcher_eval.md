# Matcher evaluation (generated)

Train calls: 6840, test calls: 1745 (temporal split, test sellers excluded from train).
Commit threshold τ = 0.6 (tunable). Labels = full-call segments from clustering (see caveat in docs).

| seller turns seen | macro-F1 | accuracy | ECE | coverage (p ≥ τ) | accuracy when covered |
|---|---|---|---|---|---|
| 1 | 0.324 | 0.592 | 0.056 | 88.2% | 0.601 |
| 2 | 0.539 | 0.679 | 0.045 | 97.8% | 0.688 |
| 3 | 0.748 | 0.801 | 0.036 | 99.3% | 0.807 |
| 4 | 0.843 | 0.869 | 0.082 | 99.9% | 0.870 |
| 5 | 0.891 | 0.905 | 0.102 | 99.9% | 0.906 |

## Per-segment precision / recall at 5 seller turns

```
              precision    recall  f1-score   support

           0      0.864     1.000     0.927       969
           1      1.000     0.781     0.877       483
           2      0.951     0.799     0.868       293

    accuracy                          0.905      1745
   macro avg      0.938     0.860     0.891      1745
weighted avg      0.916     0.905     0.903      1745

```

## Confusion matrix (rows = true segment, cols = predicted)

```
     0    1    2
0  969    0    0
1   94  377   12
2   59    0  234
```

## Turns to first confident decision

- 1 seller turns: 88.2%
- 2 seller turns: 9.6%
- 3 seller turns: 1.4%
- 4 seller turns: 0.7%
- 5 seller turns: 0.1%
- never within 5 turns (stays neutral): 0.0%

## Breakdown at 5 seller turns (look for groups where the matcher is much worse)

| group | value | calls | accuracy | coverage (p ≥ τ) |
|---|---|---|---|---|
| language_hint | __unknown__ | 1745 | 0.905 | 99.9% |
| call_objective | fix_meeting | 1745 | 0.905 | 99.9% |
| campaign_id | arrowhead | 345 | 0.829 | 100.0% |
| campaign_id | main_vani | 1400 | 0.924 | 99.9% |
