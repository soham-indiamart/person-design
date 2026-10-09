"""Persona generator: one seller-file row in, one persona spec out.

Every setting carries a value, a short reason tied to the seller's data, the evidence
key it rests on (see ``evidence.py`` / docs/findings/ps04_evidence.md), the evidence
strength, and a confidence. Missing seller fields fall back to defaults marked low
confidence; live adaptation corrects them during the call.

Personas change delivery only. ``FACTS`` and ``GUARDRAILS`` are identical for every
seller and are never edited by the generator or by live adaptation.
"""

from __future__ import annotations

import math
import re
from typing import Any

SPEC_VERSION = "ps04-1.0.0"

# What today's VANI already says on calls (ps07_call_turns, recurring bot lines).
# Personas may rephrase these; they may not add to or change them.
FACTS = {
    "agent_name": "Payal",
    "company": "IndiaMART",
    "purpose": "An IndiaMART sales executive would like a short meeting with the seller about growing their business on IndiaMART.",
    "meeting_modes": "In-person visit, or an online / phone meeting if the seller prefers.",
    "meeting_hours": "Meetings can be scheduled between 11 am and 5 pm.",
    "callback": "If the seller is busy, Payal can call back at a time they choose.",
    "details_owner": "The sales executive explains services, plans and pricing in the meeting.",
}

GUARDRAILS = [
    "Never state or estimate prices, discounts or plan contents; the executive covers them (evidence: price_inconsistency).",
    "Never invent facts, offers, enquiry numbers or deadlines beyond FACTS and the seller's own data.",
    "If the seller asks whether this is a bot, answer truthfully: Payal is IndiaMART's AI assistant.",
    "If the seller asks not to be called again, apologise, confirm, end the call and flag do-not-call.",
    "Persona settings change voice, language and delivery only, never the call's purpose or compliance content.",
    "Do not infer anything about the seller from protected traits; regional language is offered only if the seller uses it.",
]

# Female voices Sarvam recommends across hi, te, kn, ta, mr, gu, en (bulbul:v3 voices guide).
SPEAKERS = {"calm": "priya", "crisp": "ishita", "lively": "ritu"}
TTS_LANGS = {"hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}
STATE_LANG = {
    "Gujarat": "gu-IN", "Maharashtra": "mr-IN", "West Bengal": "bn-IN", "Tamil Nadu": "ta-IN",
    "Telangana": "te-IN", "Andhra Pradesh": "te-IN", "Karnataka": "kn-IN", "Kerala": "ml-IN",
    "Odisha": "od-IN", "Punjab": "pa-IN", "Assam": "as-IN", "Goa": "kok-IN",
}
ENGLISH_FIRST_STATES = {"Tamil Nadu", "Kerala", "Karnataka"}
HIGH_TURNOVER = {"1.5 - 5 Cr", "5 - 25 Cr", "25 - 100 Cr", "100 - 500 Cr", "> 500 Cr"}
VERY_HIGH_TURNOVER = {"5 - 25 Cr", "25 - 100 Cr", "100 - 500 Cr", "> 500 Cr"}

OBJECTION_MAP = {
    "Seller Unavailability & Scheduling Constraints": "busy_now",
    "Callback / Deferred Engagement (Neutral)": "call_later",
    "Explicit Disinterest / Refusal": "not_interested",
    "Existing Engagement / Past Experience": "already_in_touch",
    "Not Ready / Needs Time / Early-Stage Seller": "not_ready",
    "Information / Trust / Privacy Concerns": "trust",
    "Call Handling / Executive Gaps": "bad_past_experience",
    "Product / Service / Business Mismatch": "not_relevant",
}
PLAYBOOK = {
    "busy_now": "Acknowledge in one line, offer a callback at a time they pick, or a 2-minute summary now.",
    "call_later": "Ask for one specific time (11 am - 5 pm window) and confirm it back; do not pitch again.",
    "not_interested": "Accept once, give one seller-specific reason the meeting helps, then offer to end politely.",
    "already_in_touch": "Acknowledge the existing executive; offer to schedule the follow-up with them instead of a new pitch.",
    "not_ready": "Normalise it; offer an online meeting that just explains options, no commitment.",
    "trust": "State identity plainly (IndiaMART, AI assistant Payal), explain why you are calling, never pressure.",
    "bad_past_experience": "Apologise for the past experience, keep it short, offer a specific executive callback.",
    "not_relevant": "Check what they sell using their category; if truly irrelevant, thank them and end.",
    "price": "Do not quote; say the executive explains plans and pricing in the meeting, then ask for a time.",
    "is_this_a_bot": "Confirm truthfully that you are IndiaMART's AI assistant, then continue in one short line.",
    "send_details": "Say the executive will share details in the meeting; offer the online option to make it easy.",
}


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _str(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    v = str(v).strip()
    return v or None


def _flag(v: Any) -> bool | None:
    x = _num(v)
    return None if x is None else x == 1


def _ev(evidence: dict, key: str) -> dict:
    e = evidence.get(key, {})
    st = e.get("stats", {})
    out = {"key": key, "strength": e.get("strength", "informed_guess")}
    if "rate_group" in st:
        out["stat"] = f"{st['rate_group']:.1%} vs {st['rate_rest']:.1%} meetings (n={st['n_group']:,})"
    return out


def _setting(value: Any, reason: str, evidence: dict, key: str | None, confidence: str) -> dict:
    return {"value": value, "reason": reason, "evidence": _ev(evidence, key) if key else None,
            "confidence": confidence}


def _parse_counts(v: Any) -> list[tuple[str, int]]:
    s = _str(v)
    if not s:
        return []
    out = []
    for part in re.split(r"[;|]", s):
        m = re.match(r"\s*(.*?)\s*\((\d+)\)\s*$", part)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def traits(row: dict) -> dict:
    """Derived seller traits, each None when the underlying field is missing."""
    gst = _num(row.get("gst_registration_year"))
    avg_sec = _num(row.get("avg_answered_call_sec"))
    answered = _num(row.get("bot_answered")) or 0
    turnover = _str(row.get("annual_turnover"))
    return {
        "state": _str(row.get("seller_state")),
        "city": _str(row.get("seller_city")),
        "category": _str(row.get("top_category_1")),
        "category_group": _str(row.get("top_category_group")),
        "nature": _str(row.get("nature_of_business")),
        "business_type": _str(row.get("business_type")),
        "turnover": turnover,
        "new_seller": None if gst is None else gst >= 2024,
        "high_turnover": None if turnover is None else turnover in HIGH_TURNOVER,
        "very_high_turnover": None if turnover is None else turnover in VERY_HIGH_TURNOVER,
        "has_history": answered > 0,
        "short_calls": None if (avg_sec is None or answered == 0) else avg_sec < 25,
        "avg_call_sec": avg_sec,
        "prior_call_later": (_num(row.get("calls_call_later")) or 0) > 0,
        "prior_not_interested": (_num(row.get("calls_not_interested")) or 0) > 0,
        "repeat_not_interested": (_num(row.get("calls_not_interested")) or 0) >= 2,
        "prior_meeting": (_num(row.get("calls_meeting_fixed")) or 0) > 0,
        "frustration": _flag(row.get("showed_frustration")),
        "asked_if_bot": _flag(row.get("asked_if_talking_to_bot")),
        "in_touch": _flag(row.get("already_in_touch_with_executive")),
        "dnc": _flag(row.get("do_not_call_requested")),
        "enquiries_90d": _num(row.get("eng_enq_received_90d")),
        "cqs": _num(row.get("eng_cqs")),
        "objections": [OBJECTION_MAP[c] for c, _ in _parse_counts(row.get("past_objections")) if c in OBJECTION_MAP],
        "questions": [c for c, _ in _parse_counts(row.get("past_questions_asked"))],
    }


def seller_type(t: dict) -> str:
    """Human-readable type, used to explain the logic. The spec itself is per seller."""
    if t["dnc"]:
        return "do_not_call"
    if t["frustration"] or t["repeat_not_interested"]:
        return "guarded"
    if t["in_touch"]:
        return "already_engaged"
    if t["short_calls"] or t["prior_call_later"]:
        return "time_pressed"
    if t["very_high_turnover"] or t["business_type"] == "Limited Company":
        return "established_formal"
    if t["new_seller"]:
        return "new_receptive"
    return "standard"


def generate(row: dict, evidence: dict) -> dict:
    t = traits(row)
    stype = seller_type(t)
    missing = [k for k in ("state", "turnover", "business_type", "category", "new_seller") if t[k] is None]
    if not t["has_history"]:
        missing.append("call_history")

    if t["dnc"]:
        return {
            "seller_id": row.get("fk_glusr_usr_id"), "spec_version": SPEC_VERSION, "seller_type": stype,
            "call_policy": _setting("suppress", "The seller asked not to be called; no persona is generated.",
                                    evidence, "do_not_call", "high"),
        }

    # ---- tone --------------------------------------------------------------------------
    if stype == "guarded":
        warmth = _setting("medium", "Past frustration or repeated refusals: stay calm, not chirpy.",
                          evidence, "past_frustration" if t["frustration"] else "prior_not_interested", "medium")
        empathy = _setting("high", "Acknowledge their time and past experience before asking for anything.",
                           evidence, "past_frustration" if t["frustration"] else "prior_not_interested", "medium")
        energy = "calm"
    elif stype == "time_pressed":
        warmth = _setting("medium", "Earlier calls were short or deferred: polite but brisk.",
                          evidence, "short_call_history" if t["short_calls"] else "prior_call_later", "high")
        empathy = _setting("medium", "Acknowledge they are busy once, then move on.", evidence, "short_call_history", "medium")
        energy = "crisp"
    elif stype == "established_formal":
        warmth = _setting("medium", "Larger / incorporated business: professional, respectful register.",
                          evidence, "high_turnover" if t["very_high_turnover"] else "limited_company", "medium")
        empathy = _setting("medium", "Respect their time; lead with business value.", evidence, "high_turnover", "medium")
        energy = "crisp"
    elif stype == "already_engaged":
        warmth = _setting("high", "Already working with IndiaMART: treat as a known partner.", evidence, "already_in_touch", "high")
        empathy = _setting("high", "Acknowledge the existing executive relationship first.", evidence, "already_in_touch", "high")
        energy = "calm"
    elif stype == "new_receptive":
        warmth = _setting("high", "Newly registered business; this group fixes meetings most often: warm and encouraging.",
                          evidence, "new_seller", "high")
        empathy = _setting("medium", "Explain the benefit simply; new sellers may not know how IndiaMART helps.",
                           evidence, "new_seller", "medium")
        energy = "lively"
    else:
        warmth = _setting("medium", "No strong signal in the seller data: today's default warmth.", evidence, None, "low")
        empathy = _setting("medium", "Default; live adaptation adjusts on frustration or confusion.", evidence, None, "low")
        energy = "calm"

    # ---- language ------------------------------------------------------------------------
    lang_state = evidence.get("language_by_state", {}).get("stats", {}).get("by_state", {})
    st_share = lang_state.get(t["state"] or "", {}).get("english_leaning")
    if t["state"] in ENGLISH_FIRST_STATES:
        has_data = st_share is not None
        primary = _setting(
            "en-IN",
            f"{t['state']}: " + (f"{st_share:.0%} of past sellers here spoke mostly English vs ~2% elsewhere."
                                 if has_data else "southern state with little Hindi in past calls; English is the safer start."),
            evidence, "language_by_state", "medium" if has_data else "low")
        mix = _setting("english_heavy", "English sentences with a few common Hindi words.", evidence, "language_by_state",
                       "medium" if has_data else "low")
    elif t["state"] is None:
        primary = _setting("hi-IN", "State missing: Hinglish is what 98% of past sellers spoke.", evidence, "language_by_state", "low")
        mix = _setting("hindi_heavy", "Default Hinglish.", evidence, "language_by_state", "low")
    else:
        primary = _setting("hi-IN", "Hinglish: ~98% of past sellers spoke mostly Hindi/Hinglish.", evidence, "language_by_state", "high")
        balanced = (st_share or 0) >= 0.02 or t["very_high_turnover"] or t["business_type"] == "Limited Company"
        mix = _setting(
            "balanced" if balanced else "hindi_heavy",
            ("More English business words: " + (f"{st_share:.1%} English-leaning sellers in {t['state']}"
                                                if (st_share or 0) >= 0.02 else "larger / incorporated business"))
            if balanced else f"Hindi-led Hinglish, as most sellers in {t['state']} spoke.",
            evidence, "language_by_state" if (st_share or 0) >= 0.02 else
            ("language_by_business_type" if balanced else "language_by_state"), "medium")
    formal = bool(t["very_high_turnover"] or t["business_type"] == "Limited Company")
    formality = _setting(
        "formal" if formal else "respectful_casual",
        "Incorporated or 5 Cr+ business: 'sir/ma'am', complete sentences, business vocabulary." if formal
        else "Proprietor-style conversation: 'aap', 'ji', simple everyday words.",
        evidence, ("limited_company" if t["business_type"] == "Limited Company" else "high_turnover") if formal else None,
        "medium" if t["business_type"] is not None else "low")
    regional = STATE_LANG.get(t["state"] or "")
    switch_to = _setting(
        regional,
        (f"Prepared, not opened with: switch to {regional} only if the seller speaks it "
         f"({'voice available' if regional in TTS_LANGS else 'understood by Saaras; reply in Hindi/English, no TTS voice'})."
         if regional else "No regional language prepared; live detection still follows the seller."),
        evidence, "language_by_state", "low")

    # ---- voice ---------------------------------------------------------------------------
    pace_val = {"calm": 0.92, "crisp": 1.12, "lively": 1.02}[energy]
    if t["short_calls"] and energy != "calm":
        pace_val = max(pace_val, 1.12)
    if t["frustration"]:
        pace_val = 0.9
    pace_reason = {
        0.9: "Slower and steady: this seller showed frustration before.",
        0.92: "Slightly slower, unhurried delivery.",
        1.12: "Brisker: this seller's past calls were short / they are likely busy.",
        1.02: "Natural, upbeat pace.",
    }[pace_val]
    pace_key = ("past_frustration" if t["frustration"] else "short_call_history" if t["short_calls"]
                else "high_turnover" if energy == "crisp" else None)
    voice = {
        "gender": _setting("female", "Continuity with VANI's existing identity (Payal); no past data compares voice genders.",
                           evidence, "voice_gender", "low"),
        "speaker": _setting(SPEAKERS[energy], f"{energy} delivery; one of Sarvam's top-rated voices across Indian languages.",
                            evidence, "voice_speaker", "low"),
        "accent": _setting(primary["value"], "Accent follows the speaking language (Indian English or Hindi).",
                           evidence, "language_by_state", primary["confidence"]),
        "pitch": _setting({"calm": "lower", "crisp": "neutral", "lively": "higher"}[energy],
                          "Voice register for the tone. Sarvam bulbul:v3 has no pitch parameter (v2 is retired), "
                          "so the register is carried by the speaker choice.",
                          evidence, "pitch", "low"),
        "pace": _setting(pace_val, pace_reason, evidence, pace_key, "medium" if pace_key else "low"),
        "expressiveness": _setting({"calm": 0.45, "crisp": 0.5, "lively": 0.75}[energy],
                                   "Sarvam bulbul:v3 temperature: more expressive for warm personas.",
                                   evidence, None, "low"),
    }

    # ---- conversation plan -----------------------------------------------------------------
    opening_parts, open_keys = ["Check the line in one short phrase."], ["identity_drop"]
    if t["asked_if_bot"]:
        opening_parts.append("Introduce as IndiaMART's AI assistant Payal (they asked before).")
        open_keys.append("asked_if_bot")
    else:
        opening_parts.append("Introduce as Payal from IndiaMART.")
    if t["in_touch"]:
        opening_parts.append("Acknowledge they already work with an IndiaMART executive.")
        open_keys.append("already_in_touch")
    if t["prior_call_later"]:
        opening_parts.append("Mention they asked for a callback last time.")
        open_keys.append("prior_call_later")
    if t["frustration"]:
        opening_parts.append("Promise to keep it short.")
        open_keys.append("past_frustration")
    if t["category"]:
        opening_parts.append(f"Tie the purpose to their {t['category']} business in the same sentence.")
        open_keys.append("personalised_trial")
    else:
        opening_parts.append("State the purpose and benefit in the same sentence.")
    if stype == "time_pressed":
        opening_parts.append("Ask for the meeting time within the first two turns.")
        open_keys.append("short_call_history")
    opening = {"steps": opening_parts, "evidence": [_ev(evidence, k) for k in open_keys],
               "reason": "17% of calls drop right after the identity line: keep it short and give the reason immediately."}

    hooks = []
    if t["category"]:
        hooks.append(_setting(f"Mention their category: {t['category']}", "Personalised trial did better (weak evidence).",
                              evidence, "personalised_trial", "low"))
    if t["city"]:
        hooks.append(_setting(f"Offer an executive visit in {t['city']}", "Visits are VANI's main meeting type; the city is known.",
                              evidence, None, "medium"))
    if t["nature"]:
        hooks.append(_setting(f"Don't ask whether they manufacture or trade: already known ({t['nature']})",
                              "Today's VANI asks this; skipping it saves a turn.", evidence, None, "high"))
    if t["enquiries_90d"] and t["enquiries_90d"] > 0:
        hooks.append(_setting(f"Refer to the buyer enquiries they received recently ({int(t['enquiries_90d'])} in 90 days)",
                              "Concrete, seller-specific value from their own data.", evidence, None, "medium"))
    if t["cqs"] is not None and t["cqs"] < 40:
        hooks.append(_setting("Offer help improving their catalog", f"Catalog quality score is {int(t['cqs'])}/100.",
                              evidence, None, "medium"))

    order: list[str] = []
    for o in t["objections"]:
        if o not in order:
            order.append(o)
    priors = {
        "guarded": ["bad_past_experience", "not_interested", "trust"],
        "time_pressed": ["busy_now", "call_later"],
        "already_engaged": ["already_in_touch"],
        "established_formal": ["send_details", "price", "busy_now"],
        "new_receptive": ["not_ready", "price", "trust"],
        "standard": ["busy_now", "not_interested"],
    }[stype]
    for o in priors + ["busy_now", "call_later", "not_interested", "price", "is_this_a_bot"]:
        if o not in order:
            order.append(o)
    playbook = []
    for o in order[:6]:
        from_history = o in t["objections"]
        playbook.append({
            "objection": o, "approach": PLAYBOOK[o],
            "reason": "Raised by this seller on a past call." if from_history
            else f"Common for {stype.replace('_', ' ')} sellers." if o in priors else "Common across all sellers.",
            "evidence": _ev(evidence, "price_inconsistency" if o == "price" else
                            {"busy_now": "short_call_history", "call_later": "prior_call_later",
                             "not_interested": "prior_not_interested", "already_in_touch": "already_in_touch",
                             "bad_past_experience": "past_frustration", "trust": "asked_if_bot",
                             "is_this_a_bot": "asked_if_bot", "not_ready": "new_seller",
                             "send_details": "high_turnover", "not_relevant": "first_call"}[o]),
            "confidence": "high" if from_history else "medium",
        })

    max_refusals = 1 if (t["prior_not_interested"] or t["frustration"]) else 2
    escalation = [
        {"when": f"Seller refuses the meeting {max_refusals}x", "do": "Thank them and end politely; no further pitch.",
         "reason": "Repeat 'not interested' sellers rarely convert." if max_refusals == 1 else "Default limit.",
         "evidence": _ev(evidence, "prior_not_interested")},
        {"when": "Frustration detected twice in a row", "do": "Apologise, offer a callback at their time, end the call.",
         "evidence": _ev(evidence, "past_frustration")},
        {"when": "Asks for a human / the executive", "do": "Offer an executive callback within 11 am - 5 pm and confirm the time.",
         "evidence": _ev(evidence, "already_in_touch")},
        {"when": "Asks about price, plans or contracts", "do": "Do not quote; the executive will cover it in the meeting.",
         "evidence": _ev(evidence, "price_inconsistency")},
        {"when": "Asks not to be called", "do": "Apologise, confirm, end; flag do-not-call.", "evidence": _ev(evidence, "do_not_call")},
        {"when": "Speaks a language without a Sarvam voice", "do": "Reply in Hindi or English, whichever they understand.",
         "evidence": None},
    ]

    return {
        "seller_id": row.get("fk_glusr_usr_id"),
        "spec_version": SPEC_VERSION,
        "seller_type": stype,
        "summary": _summary(t, stype),
        "voice": voice,
        "language": {"primary": primary, "hinglish_mix": mix, "formality": formality, "switch_candidate": switch_to},
        "tone": {"warmth": warmth, "empathy": empathy},
        "conversation_plan": {
            "opening": opening, "personalisation": hooks, "objection_playbook": playbook,
            "escalation_rules": escalation, "guardrails": GUARDRAILS,
        },
        "facts": FACTS,
        "missing_fields": missing,
        "overall_confidence": "low" if len(missing) >= 3 else "medium" if missing else "high",
    }


def _summary(t: dict, stype: str) -> str:
    bits = [t["business_type"] or "unknown business type", t["nature"] or "", t["turnover"] or "turnover unknown",
            t["state"] or "state unknown"]
    hist = ("no answered calls yet" if not t["has_history"] else
            f"past calls avg {t['avg_call_sec']:.0f}s" if t["avg_call_sec"] else "some call history")
    return f"{stype.replace('_', ' ')}: " + ", ".join(b for b in bits if b) + f"; {hist}"
