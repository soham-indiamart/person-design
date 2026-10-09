from vani_persona.canonical import Speaker, Turn
from vani_persona.features.registry import FEATURES, language_features, segment_features
from vani_persona.features.transcript import extract_features, turn_signals
from vani_persona.matching.early_features import LiveFeatureState


def T(i, spk, text, s=None, e=None, **kw):
    return Turn(idx=i, speaker=Speaker(spk), text=text, start_ms=s, end_ms=e, **kw)


def test_lexicon_tags_hinglish_and_devanagari():
    assert {"brevity_request", "time_pressure"} <= turn_signals("bhai jaldi bolo, busy hoon").tags
    assert "clarification" in turn_signals("क्या बोला? समझ नहीं आया").tags
    assert "objection_trust" in turn_signals("ye fraud to nahi hai").tags
    assert turn_signals("please speak in english").language_request == "en"
    assert "opt_out" in turn_signals("dobara call mat karna").tags


def test_no_false_match_inside_words():
    # "fee" must not fire inside "feedback"; "ok" must not fire inside "book"
    tags = turn_signals("feedback book").tags
    assert "objection_price" not in tags and "affirmation" not in tags


def test_unobservable_features_are_none_not_zero():
    f = extract_features([T(0, "agent", "namaste"), T(1, "seller", "haan ji")])
    assert f["response_latency_median_ms"] is None and f["overlap_rate"] is None
    assert f["engagement_slope"] is None  # needs >= 3 seller turns
    assert f["short_reply_ratio"] == 1.0


def test_no_seller_turns():
    f = extract_features([T(0, "agent", "namaste")])
    assert f["n_seller_turns"] == 0 and f["seller_mean_tokens"] is None


def test_timing_features_from_timestamps():
    turns = [T(0, "agent", "a", 0, 1000), T(1, "seller", "b", 1400, 1500),
             T(2, "agent", "c", 1600, 3000), T(3, "seller", "d", 2800, 3200)]
    f = extract_features(turns)
    assert f["overlap_rate"] == 0.5
    assert f["response_latency_median_ms"] == 200  # median of [400, 0]


def test_registry_roles():
    assert set(language_features()).isdisjoint(segment_features())
    assert "duration_s" not in segment_features()  # short call != impatient
    assert all(f.availability in {"historical", "live", "both"} for f in FEATURES.values())


def test_live_state_window_and_low_confidence():
    st = LiveFeatureState(min_asr_confidence=0.6)
    for i, txt in enumerate(["kya fayda hoga is se aur kaise karna hai batao", "aur kitna time lagega iske liye",
                             "haan", "ok", "theek"]):
        st.add_turn(T(2 * i, "agent", "x"))
        st.add_turn(T(2 * i + 1, "seller", txt))
    assert st.features()["n_seller_turns"] == 5
    recent = st.features(recent_seller_turns=3)
    assert recent["n_seller_turns"] == 3 and recent["short_reply_ratio"] == 1.0
    assert st.add_turn(T(20, "seller", "garbled", asr_confidence=0.3)) is False
    assert st.low_confidence_turns == 1 and st.features()["n_seller_turns"] == 5


def test_filler_matlab_is_not_clarification():
    # PS07 finding: bare "matlab" is a filler; only question forms count
    assert "clarification" not in turn_signals("matlab ek dedh baje ke kareeb").tags
    assert "clarification" in turn_signals("matlab?").tags
    assert "clarification" in turn_signals("iska kya matlab hai").tags
