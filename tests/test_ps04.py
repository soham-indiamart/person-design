"""PS04 generator and live adaptation (no network: the LLM detector is switched off)."""

from vani_persona.ps04.generator import FACTS, GUARDRAILS, generate, seller_type, traits
from vani_persona.ps04.analyst import Verdict, analyse
from vani_persona.ps04.director import PersonaDirector
from vani_persona.ps04.live import LiveSettings, apply_language, apply_persona
from vani_persona.ps04.live_personas import PERSONAS, starting_persona
from vani_persona.ps04.signals import Detection, LanguageTracker, detect

EVIDENCE = {
    "new_seller": {"strength": "moderate", "stats": {"rate_group": 0.142, "rate_rest": 0.099, "n_group": 100351}},
    "language_by_state": {"strength": "moderate", "stats": {"by_state": {"Tamil Nadu": {"n": 208, "english_leaning": 0.33},
                                                                          "Delhi": {"n": 1242, "english_leaning": 0.023}}}},
}

BASE_ROW = {
    "fk_glusr_usr_id": 1, "seller_state": "Uttar Pradesh", "seller_city": "Aligarh", "business_type": "Proprietorship",
    "annual_turnover": "0 - 40 L", "nature_of_business": "Manufacturer", "gst_registration_year": 2019,
    "top_category_1": "Corrugated Box", "bot_answered": 3, "avg_answered_call_sec": 60, "calls_call_later": 0,
    "calls_not_interested": 0, "calls_meeting_fixed": 0, "showed_frustration": 0, "asked_if_talking_to_bot": 0,
    "already_in_touch_with_executive": 0, "do_not_call_requested": 0,
}


def row(**kw):
    return {**BASE_ROW, **kw}


def test_every_setting_has_reason_and_confidence():
    spec = generate(row(), EVIDENCE)
    for group in ("voice", "language", "tone"):
        for name, s in spec[group].items():
            assert s["reason"], (group, name)
            assert s["confidence"] in {"high", "medium", "low"}
    plan = spec["conversation_plan"]
    assert plan["opening"]["steps"] and plan["objection_playbook"] and plan["escalation_rules"]
    assert plan["guardrails"] == GUARDRAILS and spec["facts"] == FACTS


def test_contrasting_sellers_get_different_personas():
    new = generate(row(gst_registration_year=2025), EVIDENCE)
    rushed = generate(row(avg_answered_call_sec=9, calls_call_later=2), EVIDENCE)
    formal = generate(row(business_type="Limited Company", annual_turnover="25 - 100 Cr"), EVIDENCE)
    south = generate(row(seller_state="Tamil Nadu"), EVIDENCE)
    assert [s["seller_type"] for s in (new, rushed, formal)] == ["new_receptive", "time_pressed", "established_formal"]
    assert new["voice"]["speaker"]["value"] != rushed["voice"]["speaker"]["value"]
    assert rushed["voice"]["pace"]["value"] > new["voice"]["pace"]["value"]
    assert formal["language"]["formality"]["value"] == "formal"
    assert south["language"]["primary"]["value"] == "en-IN"
    assert south["language"]["switch_candidate"]["value"] == "ta-IN"


def test_missing_fields_lower_confidence_and_dnc_suppresses():
    sparse = generate(row(seller_state=None, annual_turnover=None, business_type=None, top_category_1=None,
                          gst_registration_year=None, bot_answered=0), EVIDENCE)
    assert sparse["overall_confidence"] == "low"
    assert sparse["language"]["primary"]["confidence"] == "low"
    dnc = generate(row(do_not_call_requested=1), EVIDENCE)
    assert dnc["call_policy"]["value"] == "suppress" and "voice" not in dnc


def test_seller_history_shapes_plan():
    spec = generate(row(already_in_touch_with_executive=1, asked_if_talking_to_bot=1,
                        past_objections="Information / Trust / Privacy Concerns (2)"), EVIDENCE)
    steps = " ".join(spec["conversation_plan"]["opening"]["steps"])
    assert "AI assistant" in steps and "executive" in steps
    assert spec["conversation_plan"]["objection_playbook"][0]["objection"] == "trust"
    assert traits(row(showed_frustration=1))["frustration"] and seller_type(traits(row(showed_frustration=1))) == "guarded"


def test_lexicon_detects_all_behavioural_signals_without_llm():
    cases = {
        "arre kitni baar call karoge": "frustration",
        "samajh nahi aaya, phir se bolo": "confusion",
        "kal kitne baje aa sakte hain": "interest",
        "abhi busy hoon jaldi bolo": "rush",
        "मतलब कौन सी मीटिंग, कुछ समझ नहीं आया": "confusion",
    }
    for text, sig in cases.items():
        dets, _ = detect(text, text, "", use_llm=False)
        assert sig in {d.signal for d in dets}, text


def test_language_switch_hysteresis():
    tr = LanguageTracker("hi-IN")
    assert tr.update("en-IN", 0.9, ["ok fine"], 2) is None  # one English sentence inside Hinglish: wait
    d = tr.update("en-IN", 0.9, ["please tell me in detail"], 5)
    assert d and d.detail == {"from": "hi-IN", "to": "en-IN"}
    tr2 = LanguageTracker("hi-IN")
    d2 = tr2.update("ta-IN", 0.88, ["எனக்கு தமிழில் பேசுங்கள்"], 3)  # regional language: immediate
    assert d2 and d2.detail["to"] == "ta-IN"
    tr3 = LanguageTracker("hi-IN")
    assert tr3.update(None, None, ["english mein bolo"], 3).detail["to"] == "en-IN"  # explicit request


def _settings(**kw):
    base = dict(language="hi-IN", mix="balanced", formality="respectful_casual", speaker="ishita", pace=1.1,
                expressiveness=0.5, warmth="medium", empathy="medium")
    return LiveSettings(**{**base, **kw})


def _v(state, intensity=0.8, trend="steady", source="analyst", lexicon=None):
    return Verdict(state, intensity, trend, "", "", source, lexicon or {})


def test_director_switches_on_clear_signals_and_recovers():
    d = PersonaDirector("normal")
    assert d.decide(_v("calm", 0.3)).action == "stay"
    sw = d.decide(_v("aggressive", 0.8, "escalating"))
    assert (sw.action, sw.to_persona) == ("switch", "de_escalator")
    assert d.decide(_v("calm", 0.4, "calming")).action == "stay"       # one calm turn is not enough
    back = d.decide(_v("calm", 0.3, "calming"))
    assert (back.action, back.to_persona) == ("switch", "normal")
    assert d.decide(_v("interested", 0.75)).to_persona == "closer"


def test_director_weak_signals_need_repeat_or_agreement():
    d = PersonaDirector("normal")
    assert d.decide(_v("confused", 0.55)).action == "pending"
    assert d.decide(_v("confused", 0.55)).to_persona == "patient_explainer"
    d2 = PersonaDirector("normal")
    agreed = _v("confused", 0.5, source="analyst+lexicon", lexicon={"confused": "samajh nahi"})
    assert d2.decide(agreed).to_persona == "patient_explainer"
    d3 = PersonaDirector("normal")
    assert d3.decide(_v("rushed", 0.3)).action == "stay"               # too faint


def test_director_aggression_is_urgent_and_escalates():
    d = PersonaDirector("closer")
    sw = d.decide(_v("aggressive", 0.4, lexicon={"aggressive": "bakwas"}))  # phrase hit beats low intensity
    assert sw.to_persona == "de_escalator"
    assert d.decide(_v("aggressive", 0.8)).action == "stay"
    assert d.decide(_v("aggressive", 0.9)).action == "escalate"          # 3rd aggressive turn in a row


def test_director_blocks_ping_pong():
    d = PersonaDirector("normal")
    d.decide(_v("rushed", 0.75))                                          # normal -> quick_direct
    d.decide(_v("confused", 0.75))                                        # quick_direct -> patient_explainer
    assert not d.decide(_v("rushed", 0.72)).switched                      # straight back needs >= 0.8
    assert d.decide(_v("rushed", 0.85)).to_persona == "quick_direct"


def test_persona_templates_change_delivery_only():
    base = _settings()
    for pid in PERSONAS:
        s = _settings()
        changes = apply_persona(s, base, pid)
        assert (s.language, s.mix, s.formality, s.speaker) == (base.language, base.mix, base.formality, base.speaker)
        assert set(changes) <= {"pace", "expressiveness", "warmth", "empathy", "sentence_length", "vocabulary"}
    s = _settings()
    apply_persona(s, base, "de_escalator")
    apply_persona(s, base, "normal")
    assert s.pace == base.pace and s.warmth == base.warmth              # back to the seller baseline


def test_starting_persona_from_seller_file():
    assert starting_persona(generate(row(avg_answered_call_sec=9), EVIDENCE))[0] == "quick_direct"
    assert starting_persona(generate(row(asked_if_talking_to_bot=1), EVIDENCE))[0] == "reassurer"
    assert starting_persona(generate(row(), EVIDENCE))[0] == "normal"


def test_analyst_falls_back_to_lexicon(monkeypatch):
    from vani_persona.ps04 import sarvam

    def boom(*a, **k):
        raise sarvam.SarvamError("offline")

    monkeypatch.setattr(sarvam, "chat", boom)
    v = analyse([("bot", "Namaste"), ("seller", "kitni baar call karoge, bakwas")], "", "normal")
    assert (v.state, v.source) == ("aggressive", "lexicon")
    v = analyse([("bot", "Namaste"), ("seller", "haan ji bolo")], "", "normal")
    assert v.state == "calm"


def test_language_mirroring_is_independent_of_persona():
    base, s = _settings(), _settings()
    apply_persona(s, base, "closer")
    apply_language(s, base, Detection("language_switch", 0.9, "saaras", "", {"from": "hi-IN", "to": "as-IN"}))
    assert s.language == "hi-IN"  # Assamese is understood but has no Sarvam voice: stay in Hindi
    apply_language(s, base, Detection("language_switch", 0.9, "saaras", "", {"from": "hi-IN", "to": "ta-IN"}))
    assert (s.language, s.mix, s.warmth) == ("ta-IN", "native", "high")  # persona delivery untouched


def test_credit_cap_blocks_before_sending(tmp_path):
    import pytest

    from vani_persona.ps04.sarvam import Budget, CreditLimitReached, stt_cost, tts_cost

    b = Budget(tmp_path / "usage.json", limit=1.0)
    b.record("text_to_speech", tts_cost("x" * 300))  # 0.9 credits
    assert b.status()["remaining"] == 0.1
    with pytest.raises(CreditLimitReached):
        b.check("text_to_speech", tts_cost("x" * 100))  # 0.3 more would pass the cap
    b.check("speech_to_text", stt_cost(b"\x00" * 16000))  # ~1 s of audio is cheap: allowed
