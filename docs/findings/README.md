# Findings (PS07 export)

| File | Produced by | Contents |
|---|---|---|
| `discovery_log.md` | hand-written from the reports | Every run, decision and caveat on the way to the persona library: **start here** |
| `dataset_overview.md` | `python -m vani_persona.analysis.ps07_overview .` | Every table of the export: baseline, ICC, attempts, hours, evaluators, latency, language, executive calls |
| `data_audit.md` | `python -m vani_persona.audit . --source ps07` | Coverage, outcomes, confounders, feature availability |
| `segmentation_report.md` | `python -m vani_persona.segmentation.pipeline . --source ps07` | Candidate k, validation gates, segments, proxy check (run `seg-63b6bc47e6`) |
| `outcome_association.md` | same | Confounder-adjusted meeting rate per segment (descriptive) |
| `matcher_eval.md` | `python -m vani_persona.matching.train . --source ps07` | Early-turn matcher quality on a temporal holdout |
| `replay_report.md` | `python -m vani_persona.matching.replay . --source ps07` | Held-out calls replayed through the live session |

All files contain aggregates only. Regenerate them with `VANI_PSEUDONYM_SALT` set; segments and metrics do not depend on the salt.
