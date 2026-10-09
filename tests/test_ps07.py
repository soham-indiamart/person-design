"""PS07 loader on a tiny SYNTHETIC multi-table fixture (same columns as the real export, invented rows)."""

import json

import pandas as pd

from vani_persona.canonical import DispositionClass, Speaker
from vani_persona.ingest.ps07 import load_ps07


def write_fixture(root):
    d1, d2, d3 = (root / f"Dataset {i} of 3" for i in (1, 2, 3))
    for d in (d1, d2, d3):
        d.mkdir()
    pd.DataFrame(
        [
            (1, 100, 0, "bot", 0.0, 3.0, "Namaste, main VANI IndiaMART se"),
            (1, 100, 1, "seller", 3.0, 4.0, "haan boliye, mera number 9876543210 hai"),
            (2, 200, 0, "bot", 0.0, 2.0, "Namaste"),
            (2, 200, 1, "seller", 2.5, 3.0, "galat number"),
            (3, 300, 0, "bot", 0.0, 2.0, "Namaste"),  # no bot_calls row -> skipped
        ],
        columns=["attempt_id", "fk_glusr_usr_id", "turn_no", "speaker", "start_sec", "end_sec", "text"],
    ).to_csv(d1 / "ps07_call_turns.csv", index=False)
    pd.DataFrame(
        [(100, "Textiles", "Gujarat"), (200, "Food", "Delhi")],
        columns=["fk_glusr_usr_id", "top_category_group", "seller_state"],
    ).to_csv(d1 / "ps07_seller_files_part1.csv", index=False)
    pd.DataFrame(
        [
            (1, 100, "PIM", "main_vani", 1, "2026-07-10 11:00:00", 60, "s", "u", "Meeting Fixed", 1, 1, 1),
            (2, 200, "UA", "arrowhead", 1, "2026-07-11 12:00:00", 20, "s", "u", "General (talked)", 0, 1, 1),
        ],
        columns=["attempt_id", "fk_glusr_usr_id", "redis_bucket", "lead_bot_version", "call_attempt_count",
                 "call_start_time", "lead_call_duration", "lead_call_summary", "call_recording_url",
                 "disposition_label", "meeting_fixed", "has_turn_transcript", "has_evaluator_output"],
    ).to_csv(d2 / "ps07_bot_calls_2026-07.csv", index=False)
    pd.DataFrame(
        [
            (1, 100, "incorrect_mf", json.dumps({"incorrect_mf": True})),
            (2, 200, "disposition", json.dumps({"disposition": "wrong_number", "sub_disposition": "none"})),
        ],
        columns=["attempt_id", "fk_glusr_usr_id", "eval_agent", "agent_output"],
    ).to_csv(d3 / "ps07_evaluator_outputs_2026-07.csv", index=False)


def test_load_ps07_joins_and_maps(tmp_path):
    write_fixture(tmp_path)
    calls = {c.call_id: c for c in load_ps07(tmp_path)}
    assert set(calls) == {"1", "2"}  # call 3 has no bot_calls row
    c1, c2 = calls["1"], calls["2"]
    assert [t.speaker for t in c1.turns] == [Speaker.AGENT, Speaker.SELLER]
    assert c1.turns[1].start_ms == 3000 and "<PHONE>" in c1.turns[1].text
    assert c1.context.campaign_id == "main_vani" and c1.context.lead_type == "PIM"
    assert c1.context.seller_category == "Textiles" and c1.context.started_at.utcoffset().total_seconds() == 19800
    # meeting_fixed=1 but flagged by incorrect_mf -> not achieved
    assert c1.outcome.objective_achieved is False
    assert c2.outcome.disposition_class == DispositionClass.WRONG_PARTY
    assert c1.seller_key and c1.seller_key != "100"
