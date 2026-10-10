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


def test_pace_requests_and_script_detection():
    from vani_persona.ps04.signals import pace_request, script_language

    assert pace_request(["bhai itna dheere kyu bol rahe ho"])[0] == "faster"
    assert pace_request(["इतना धीरे बोल रहे हो, समझ नहीं आ रहा"])[0] == "faster"
    assert pace_request(["thoda dheere boliye please"])[0] == "slower"
    assert pace_request(["haan ji bolo"])[0] is None
    assert script_language("ইয়ার আবি বি ইতনা স্লো")[0] == "bn-IN"
    assert script_language("haan ji bolo")[0] is None


def test_language_switches_on_first_full_sentence():
    tr = LanguageTracker("hi-IN")
    d = tr.update(None, None, ["ইয়ার এত স্লো বললে চলবে না"], 5)  # Bengali script, no Saaras code (typed)
    assert d and d.detail["to"] == "bn-IN"
    tr2 = LanguageTracker("hi-IN")
    d2 = tr2.update(None, None, ["Okay, tell me quickly what this is about"], 8, hint="en-IN")
    assert d2 and d2.detail["to"] == "en-IN"                       # analyst read alone, full sentence
    tr3 = LanguageTracker("hi-IN")
    assert tr3.update("en-IN", 0.95, ["Okay, tell me what this meeting is"], 7) is not None  # sure Saaras


def test_speed_complaint_is_rushed_not_confused(monkeypatch):
    from vani_persona.ps04 import sarvam

    monkeypatch.setattr(sarvam, "chat", lambda *a, **k: (_ for _ in ()).throw(sarvam.SarvamError("offline")))
    v = analyse([("bot", "Namaste"), ("seller", "itna dheere bol rahe ho, samajh nahi aa raha")], "", "normal")
    assert v.state == "rushed"


def test_live_profile_pace_is_sticky_and_sets_home_persona():
    from vani_persona.ps04.profile import LiveProfile, apply_profile

    p = LiveProfile(language="hi-IN", mix="balanced", formality="respectful_casual")
    assert p.home_persona == "normal"
    ch = p.observe(words=6, audio_s=None, analyst={}, pace_request="faster", language="hi-IN", mix=None, seller_turn=1)
    assert ch["pace_bias"] == [0.0, 0.12] and p.home_persona == "quick_direct"
    for _ in range(3):  # the analyst repeating "faster" does not ratchet the pace
        p.observe(words=6, audio_s=None, analyst={"pace": "faster"}, pace_request=None, language="hi-IN", mix=None,
                  seller_turn=2)
    assert p.pace_bias == 0.12
    for pid in ("de_escalator", "patient_explainer", "normal"):   # persona switches never undo it
        base, s = _settings(), _settings()
        apply_persona(s, base, pid)
        slow = s.pace
        apply_profile(s, p)
        assert s.pace == round(slow + 0.12, 2)
    p.observe(words=2, audio_s=None, analyst={"patience": "low"}, pace_request=None, language="hi-IN", mix=None,
              seller_turn=3)
    p.observe(words=8, audio_s=None, analyst={"patience": "medium"}, pace_request=None, language="hi-IN", mix=None,
              seller_turn=4)
    assert p.patience == "low"                                   # one calmer read is not enough
    p.observe(words=8, audio_s=None, analyst={"patience": "medium"}, pace_request=None, language="hi-IN", mix=None,
              seller_turn=5)
    assert p.patience == "medium"
    fast = LiveProfile(language="hi-IN", mix="balanced", formality="formal")
    fast.observe(words=12, audio_s=3.9, analyst={}, pace_request=None, language="hi-IN", mix=None, seller_turn=1)
    assert fast.speech_rate_wps == 4.0 and fast.rate_bias > 0     # a fast talker gets a brisker voice


def test_director_rests_in_profile_home_persona():
    d = PersonaDirector("normal")
    sw = d.decide(_v("calm", 0.3), home="quick_direct", home_reason="seller is impatient")
    assert (sw.action, sw.to_persona) == ("switch", "quick_direct")
    d2 = PersonaDirector("de_escalator")
    d2.decide(_v("calm", 0.3, "calming"), home="quick_direct")
    assert d2.decide(_v("calm", 0.3, "calming"), home="quick_direct").to_persona == "quick_direct"


def test_session_switches_language_pace_and_persona_offline(monkeypatch, tmp_path):
    import json as _json

    from vani_persona.ps04 import live, sarvam
    from vani_persona.ps04.live import CallSession

    monkeypatch.setattr(live, "LOG_DIR", tmp_path)
    reads = iter([
        {"state": "rushed", "intensity": 0.8, "trend": "escalating", "language": "hi-IN", "pace": "faster",
         "patience": "low", "formality": "casual", "persona": "busy shop owner, impatient"},
        {"state": "calm", "intensity": 0.4, "trend": "steady", "language": "en-IN", "pace": "same",
         "patience": "low", "formality": "casual", "persona": "busy owner, now in English"},
    ])

    def fake_chat(messages, max_tokens=300, temperature=0.3, json_mode=False, **k):
        if json_mode:
            return _json.dumps(next(reads)), 5
        return "Ji, ek chhoti meeting fix karni thi. Kab free hain?", 5

    monkeypatch.setattr(sarvam, "chat", fake_chat)
    spec = generate(row(), EVIDENCE)
    s = CallSession(spec, start_persona="normal")
    s.open(stream=True)
    start_pace = s.settings.pace
    r = s.seller_text("bhai itna dheere kyu bol rahe ho, jaldi bolo", stream=True)
    assert r["persona"]["id"] == "quick_direct"
    assert r["settings"]["pace"] > start_pace and r["profile"]["patience"] == "low"
    assert {e["type"] for e in r["switches"]} >= {"persona", "profile"}
    r = s.seller_text("Okay, tell me quickly what this meeting is about", stream=True)
    assert r["settings"]["language"] == "en-IN"
    assert any(e["type"] == "language" for e in r["switches"])
    assert r["persona"]["id"] == "quick_direct"                  # calm, but the profile says impatient


def test_looks_incomplete_waits_for_unfinished_thoughts():
    from vani_persona.ps04.signals import looks_incomplete

    assert looks_incomplete("main ye kehna chah raha tha ki")[0]
    assert looks_incomplete("umm")[0]
    assert looks_incomplete("dekhiye hum log...")[0]
    assert looks_incomplete("मुझे ये चाहिए लेकिन")[0]
    assert not looks_incomplete("haan ji")[0]
    assert not looks_incomplete("kaun bol raha hai?")[0]
    assert not looks_incomplete("abhi busy hoon, baad mein call karo")[0]


def test_new_personas_have_states_and_buyer_behaviour():
    for pid, state in [("gentle_nudger", "hesitant"), ("empathetic_listener", "complaining"),
                       ("advisor", "inquisitive"), ("respectful_persuader", "uninterested")]:
        assert PERSONAS[pid].seller_state == state
    assert all(p.behaviour_for("buyer") for p in PERSONAS.values())
    s, base = _settings(), _settings()
    apply_persona(s, base, "advisor", "buyer")
    assert s.focus == PERSONAS["advisor"].buyer_behaviour


def test_analyst_reads_complaints_and_questions_offline(monkeypatch):
    from vani_persona.ps04 import sarvam

    monkeypatch.setattr(sarvam, "chat", lambda *a, **k: (_ for _ in ()).throw(sarvam.SarvamError("offline")))
    assert analyse([("seller", "pichli baar koi enquiry nahi aayi, paisa barbaad hua")], "", "normal").state \
        == "complaining"
    assert analyse([("seller", "ye kaise kaam karta hai, kya fayda hoga?")], "", "normal").state == "inquisitive"


def test_buyer_spec_and_flow_tags(monkeypatch, tmp_path):
    from vani_persona.ps04 import live, sarvam
    from vani_persona.ps04.flows import buyer_spec, flow_for
    from vani_persona.ps04.live import CallSession

    monkeypatch.setattr(live, "LOG_DIR", tmp_path)
    spec = buyer_spec(generate(row(), EVIDENCE), "steel pipes")
    assert spec["call_mode"] == "buyer" and flow_for(spec).mode == "buyer"
    monkeypatch.setattr(sarvam, "chat", lambda *a, **k: (
        "Ji, kitni quantity chahiye? ##STAGE: quantity## ##NOTE: product=steel pipes; city=Pune##", 5))
    s = CallSession(spec, start_persona="normal")
    r = s.open(stream=True)
    assert "##" not in r["reply"] and "STAGE" not in r["reply"]
    assert r["flow"]["stage"] == "quantity" and r["flow"]["notes"] == {"product": "steel pipes", "city": "Pune"}
    assert "requirement" in r["flow"]["visited"] and "quantity" in r["flow"]["visited"]
    assert "buyer" in s.system_prompt(s.settings, s.persona_id, False).lower()


def test_session_waits_for_unfinished_turn_then_merges(monkeypatch, tmp_path):
    from vani_persona.ps04 import live, sarvam
    from vani_persona.ps04.live import CallSession

    monkeypatch.setattr(live, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sarvam, "chat", lambda *a, **k: (_ for _ in ()).throw(sarvam.SarvamError("offline"))
                        if k.get("json_mode") else ("Ji, samajh gayi. ##STAGE: discover##", 5))
    s = CallSession(generate(row(), EVIDENCE), start_persona="normal")
    s.open(stream=True)
    r = s.seller_text("main ye kehna chah raha tha ki", stream=True, hold_if_unfinished=True)
    assert r["wait"] and s.pending and not any(t.role == "seller" for t in s.turns)
    r = s.seller_text("hum log steel pipes bechte hain", stream=True, hold_if_unfinished=True)
    assert not r.get("wait") and r["transcript"] == "main ye kehna chah raha tha ki hum log steel pipes bechte hain"
    assert sum(t.role == "seller" for t in s.turns) == 1 and r["flow"]["stage"] == "discover"
    assert s.seller_text("umm", stream=True, hold_if_unfinished=True)["wait"]
    r = s.flush(stream=True)                                     # the caller went quiet: answer anyway
    assert r["transcript"] == "umm" and r["reply"]
    s.mark_interrupted()
    assert s.turns[-1].meta["interrupted"]
    r = s.nudge(stream=True)
    assert r["nudge"] == 1 and not r["ended"]


def test_cooked_seller_file_maps_every_use_case_and_archetype():
    from vani_persona.ps04.seller_file import SAMPLE, build_spec, parse
    from vani_persona.ps04.use_cases import USE_CASES

    rows = parse(SAMPLE.read_bytes(), SAMPLE.name)
    specs = {r["company_name"]: build_spec(r, EVIDENCE) for r in rows}
    assert "voice" not in specs["Kumar Sanitary Store"]                     # do-not-call: suppressed
    got = {s["use_case"]["id"] for s in specs.values() if "use_case" in s}
    assert got == set(USE_CASES)
    expect = {
        "Sharma Bartan Bhandar": ("seller_enrichment", "shopkeeper", "normal"),
        "Patel Polymers Pvt Ltd": ("seller_enrichment", "manufacturer", "advisor"),
        "Verma Electricals": ("meeting_scheduling", "rushed", "quick_direct"),
        "Shree Ganesh Steel": ("lead_qualification", "experienced", "normal"),
        "Kaveri Handlooms": ("lead_qualification", "unfamiliar", "patient_explainer"),
        "Bharat Packaging": ("response_recovery", "disengaged", "gentle_nudger"),
        "Maa Durga Sweets": ("onboarding", "beginner", "patient_explainer"),
        "Gupta Hardware & Tools": ("renewal", "dissatisfied", "empathetic_listener"),
        "Royal Furniture House": ("issue_resolution", "frustrated", "empathetic_listener"),
        "Bengal Jute Exports Ltd": ("catalogue_verification", "large", "advisor"),
    }
    for company, (uid, arch, persona) in expect.items():
        uc = specs[company]["use_case"]
        assert (uc["id"], uc["archetype"]["id"], uc["start_persona"]) == (uid, arch, persona), company
    assert specs["Shree Ganesh Steel"]["use_case"]["reason"].startswith("Inferred")   # blank use_case column
    assert any("TMT" in c for c in specs["Shree Ganesh Steel"]["use_case"]["context"])


def test_use_case_inference_priority_and_override():
    from vani_persona.ps04.use_cases import apply_use_case, infer_use_case

    needs = {"ticket_summary": "app login fails", "subscription_expiry_days": 10, "unreplied_enquiries": 4}
    assert infer_use_case(needs)[0] == "issue_resolution"                   # an open issue comes first
    assert infer_use_case({**needs, "ticket_status": "closed"})[0] == "renewal"
    assert infer_use_case({})[0] == "meeting_scheduling"
    assert infer_use_case({"use_case": "Catalogue"})[0] == "catalogue_verification"
    spec = apply_use_case(generate(row(), EVIDENCE), {"listings_to_verify": "Pipe | 10 rupees"}, "catalogue_verification")
    plan = spec["conversation_plan"]
    assert spec["call_mode"] == "catalogue_verification" and "meeting" not in spec["facts"]["purpose"].lower()
    assert not any("meeting" in s.lower() for s in plan["opening"]["steps"])
    assert not any("meeting" in p["approach"].lower() for p in plan["objection_playbook"])
    assert spec["use_case"]["checklist"] == [{"key": "listing_1", "label": "Pipe"}]


def test_use_case_session_prompt_checklist_and_rest_persona(monkeypatch, tmp_path):
    from vani_persona.ps04 import live, sarvam
    from vani_persona.ps04.live import CallSession
    from vani_persona.ps04.use_cases import apply_use_case

    monkeypatch.setattr(live, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sarvam, "chat", lambda *a, **k: (
        "Ji, aapke brands kaun se hain? ##STAGE: collect## ##NOTE: brands=Prestige, Hawkins##", 5))
    data = {"nature_of_business": "Manufacturer", "profile_missing_fields": "brands; moq"}
    spec = apply_use_case(generate(row(), EVIDENCE), data)
    s = CallSession(spec)
    assert s.persona_id == s.rest_persona == "advisor"
    prompt = s.system_prompt()
    assert "USE CASE: Seller enrichment" in prompt and "brands = brands they deal in" in prompt
    assert PERSONAS["quick_direct"].behaviour_for("seller_enrichment") != PERSONAS["quick_direct"].behaviour
    r = s.open(stream=True)
    assert r["flow"]["progress"] == {"done": 1, "total": 2} and r["flow"]["stage"] == "collect"
    assert "brands =" not in s.system_prompt() and "moq = " in s.system_prompt()   # answered items drop off


def test_credit_cap_blocks_before_sending(tmp_path):
    import pytest

    from vani_persona.ps04.sarvam import Budget, CreditLimitReached, stt_cost, tts_cost

    b = Budget(tmp_path / "usage.json", limit=1.0)
    b.record("text_to_speech", tts_cost("x" * 300))  # 0.9 credits
    assert b.status()["remaining"] == 0.1
    with pytest.raises(CreditLimitReached):
        b.check("text_to_speech", tts_cost("x" * 100))  # 0.3 more would pass the cap
    b.check("speech_to_text", stt_cost(b"\x00" * 16000))  # ~1 s of audio is cheap: allowed
