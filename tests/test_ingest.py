import pandas as pd

from vani_persona.canonical import DispositionClass, Speaker
from vani_persona.ingest.adapter import iter_calls, load_mapping, parse_transcript
from vani_persona.ingest.redact import pseudonymize, redact


def test_redact_phone_email_gstin():
    text, n = redact("call me on +91 98765 43210 or a.b@x.com, GST 27ABCDE1234F1Z5")
    assert "98765" not in text and "a.b@x.com" not in text and "27ABCDE1234F1Z5" not in text
    assert n == 3


def test_redact_keeps_short_numbers():
    text, n = redact("2 products, pincode 400001")
    assert text == "2 products, pincode 400001" and n == 0


def test_pseudonymize_is_salted_and_stable():
    assert pseudonymize("s1", "salt") == pseudonymize("s1", "salt")
    assert pseudonymize("s1", "salt") != pseudonymize("s1", "other")
    assert pseudonymize("", "salt") is None


def test_parse_transcript_prefixes_and_continuations():
    m = load_mapping()
    turns = parse_transcript("Agent: Namaste\nSeller: haan ji\nboliye\nBot: ok", m)
    assert turns == [(Speaker.AGENT, "Namaste"), (Speaker.SELLER, "haan ji boliye"), (Speaker.AGENT, "ok")]


def test_wide_layout_maps_context_outcome_and_redacts():
    m = load_mapping()
    df = pd.DataFrame(
        [{"call_id": "c1", "seller_id": "S9", "campaign_id": "k", "language": "hi", "disposition": "Not Interested",
          "call_start_time": "2026-07-01T10:00:00Z", "transcript": "Agent: hi\nSeller: mera number 9876543210 hai"}]
    )
    (call,) = list(iter_calls(df, m))
    assert call.seller_key != "S9" and call.seller_key
    assert call.context.campaign_id == "k" and call.context.started_at.hour == 10
    assert call.outcome.disposition_class == DispositionClass.NOT_INTERESTED
    assert call.outcome.objective_achieved is False
    assert "<PHONE>" in call.turns[1].text and call.quality.redactions_applied == 1


def test_long_layout_orders_turns_and_reads_timestamps():
    m = {**load_mapping(), "layout": "long"}
    df = pd.DataFrame(
        [
            {"call_id": "c1", "speaker": "seller", "text": "haan", "turn_index": "1", "start_ms": "1500", "end_ms": "1800"},
            {"call_id": "c1", "speaker": "agent", "text": "namaste", "turn_index": "0", "start_ms": "0", "end_ms": "1200"},
        ]
    )
    (call,) = list(iter_calls(df, m))
    assert [t.speaker for t in call.turns] == [Speaker.AGENT, Speaker.SELLER]
    assert call.quality.has_timestamps
