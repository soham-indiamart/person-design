"""End-to-end on SYNTHETIC calls: discovery -> matcher training -> live scoring.

Checks only that the code path works and recovers planted structure. Says nothing about real sellers.
"""

import shutil

import pandas as pd
import pytest
import yaml
from sklearn.metrics import adjusted_rand_score

from synthetic import make_rows, write_csv
from vani_persona import PERSONA_DIR
from vani_persona.matching.matcher import LogisticScorer, MatchRequest, PersonaMatcher
from vani_persona.matching.train import run as train_run
from vani_persona.personas import PersonaRegistry
from vani_persona.segmentation.pipeline import run as seg_run
from vani_persona.segmentation.preprocess import build_frame
from vani_persona.ingest.adapter import load_calls


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    rows = make_rows(600)
    write_csv(root / "raw" / "calls.csv", rows)
    ctx = seg_run(root / "raw", root / "findings", root / "artifacts", root / "derived", n_boot=20)
    train_run(root / "raw", root / "derived" / "segments.csv", root / "artifacts", root / "findings")
    styles = pd.DataFrame(rows)[["call_id", "_style"]]
    return root, ctx, styles


def test_discovery_recovers_planted_segments(built):
    root, ctx, styles = built
    assert ctx["chosen"] is not None and ctx["chosen"].k == 3
    seg = pd.read_csv(root / "derived" / "segments.csv").merge(styles, on="call_id")
    assert adjusted_rand_score(seg["_style"], seg["segment"]) > 0.9
    assert len(seg) == 600  # terse calls must not be filtered out of discovery
    report = (root / "findings" / "segmentation_report.md").read_text()
    assert "pending human interpretability review" in report and "Proxy / stereotyping check" in report
    assert (root / "findings" / "outcome_association.md").exists()


def test_reports_contain_no_transcript_text(built):
    root, _, _ = built
    for f in (root / "findings").glob("*.md"):
        text = f.read_text()
        assert "busy hoon" not in text and "fraud" not in text


def test_trained_matcher_scores_live_prefix(built, tmp_path):
    root, _, styles = built
    seg = pd.read_csv(root / "derived" / "segments.csv").merge(styles, on="call_id")
    terse_segment = int(seg[seg["_style"] == "terse"]["segment"].mode()[0])
    # Persona library whose personas point at the discovered segments.
    shutil.copy(PERSONA_DIR / "neutral_default.yaml", tmp_path)
    for i in range(3):
        p = yaml.safe_load((PERSONA_DIR / "objection_raising.yaml").read_text())
        p["persona_id"], p["status"] = f"cand_seg_{i}", "candidate"
        p["matching"] = {"segment_id": i}
        (tmp_path / f"cand_{i}.yaml").write_text(yaml.safe_dump(p))
    reg = PersonaRegistry(tmp_path)
    matcher = PersonaMatcher(LogisticScorer(root / "artifacts" / "matcher.joblib", reg), 2, 0.6)

    call = next(c for c in load_calls(root / "raw") if c.call_id in set(styles[styles._style == "terse"].call_id))
    feats = build_frame([call], max_seller_turns=2).iloc[0].to_dict()
    r = matcher.match(MatchRequest(call.call_id, 3, feats, {"campaign_id": call.context.campaign_id,
                                                          "call_objective": call.context.call_objective,
                                                          "language_hint": call.context.language_hint}))
    assert r.calibrated and r.scorer == "logistic"
    assert r.best_persona_id == f"cand_seg_{terse_segment}" and not r.abstained
    assert r.evidence and abs(sum(r.scores.values()) - 1) < 1e-6


def test_audit_report(built):
    from vani_persona.audit import audit

    root, _, _ = built
    text = audit(root / "raw")
    assert "# Data audit" in text and "Calls with turn timestamps: 0%" in text
    assert "busy hoon" not in text  # aggregates only
