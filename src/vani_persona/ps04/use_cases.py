"""VANI use cases: what a seller call is for, and how the persona fits that purpose.

A seller-file row says *who* the seller is (the generator turns that into voice, language,
tone and plan) and, through its use-case columns, *what this call needs from them*:
missing profile fields, an enquiry to qualify, unanswered enquiries, onboarding steps,
a subscription about to expire, an open complaint, listings to verify, or a paid plan
they left (win-back).

For every seller this module

1. picks the use case: the ``use_case`` column if set, otherwise inferred from the data
   (an open ticket outranks a renewal, which outranks unanswered enquiries, ...);
2. picks the seller's archetype within that use case (e.g. busy shopkeeper vs.
   manufacturer for enrichment), which sets the starting / resting live persona, the
   approach and a few delivery settings, each with its reason;
3. builds the call context (the requirement, from the seller's own data) and the
   checklist the call has to complete (the use case's success metric);
4. swaps the meeting-only facts, opening, playbook and escalation wording for the use
   case's, keeping the guardrails.

Live adaptation (analyst, director, profile) then works exactly as before on top.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from vani_persona.ps04.flows import SELLER, Flow, Stage
from vani_persona.ps04.generator import FACTS as MEETING_FACTS
from vani_persona.ps04.generator import traits


# ---- small helpers ------------------------------------------------------------------------
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


def _list(v: Any) -> list[str]:
    s = _str(v)
    return [p.strip() for p in re.split(r"[;\n]", s) if p.strip()] if s else []


def _key(text: str) -> str:
    return re.sub(r"\W+", "_", text.lower()).strip("_")[:30]


@dataclass(frozen=True)
class Archetype:
    id: str
    label: str
    persona: str            # starting and resting live persona
    approach: str           # how to run this use case with this kind of seller (goes into the prompt)
    delivery: dict = field(default_factory=dict)  # sentence_length / vocabulary / pace_delta / warmth / empathy


@dataclass(frozen=True)
class UseCase:
    id: str
    name: str
    goal: str
    metric: str
    ask: str                # the main thing the call asks for, used in playbook / escalation wording
    benefit: str            # one reason the call helps the seller
    facts: dict
    flow: Flow
    archetypes: dict
    choose: Callable[[dict, dict, str], tuple[str, str]]   # (traits, row, seller_type) -> (archetype id, reason)
    context: Callable[[dict], list[str]]                   # row -> requirement lines for the prompt
    checklist: Callable[[dict], list[tuple[str, str]]]     # row -> [(note key, what to capture)]


COMMON_HANDLERS = (
    "Who is calling / is this a bot: Payal, IndiaMART's AI assistant, calling about their IndiaMART seller account.",
    "How did you get my number: from their registered IndiaMART business details.",
    "Price, plan or payment questions: never quote; the IndiaMART team / executive confirms prices and plans.",
    "Busy: offer to finish in two minutes now or call back at a time they pick.",
    "Cannot hear / network issue: ask once whether they can hear you; if it repeats, promise a callback and close.",
    "Asks not to be called: apologise, confirm, close.",
)
GENERAL_STYLE = (
    "Sound like an experienced, friendly IndiaMART relationship manager, not a script: react to what they just said first.",
    "Use short natural acknowledgements (ji, achha, bilkul, samajh gayi, theek hai) and vary them.",
    "Never repeat back what they just told you; never re-ask something already answered or already in the call context.",
    "Ask at most one question per reply, and first answer what they asked.",
    "People speak in bits: combine their fragments with earlier turns before replying.",
)


def _facts(purpose: str, next_step: str, extra: dict | None = None) -> dict:
    return {
        "agent_name": "Payal", "company": "IndiaMART", "purpose": purpose, "next_step": next_step,
        "callback": "If the seller is busy, Payal can call back at a time they choose.",
        "details_owner": "Prices, plans and contracts are confirmed only by the IndiaMART team, never on this call.",
        **(extra or {}),
    }


def _flow(uid: str, name: str, role: str, stages: list[tuple[str, str, str]], handlers: tuple[str, ...],
          note_keys: tuple[str, ...]) -> Flow:
    return Flow(mode=uid, label=name, counterpart="seller", role=role,
                stages=tuple(Stage(*s) for s in stages), handlers=COMMON_HANDLERS + handlers,
                style=GENERAL_STYLE, note_keys=note_keys)


IDENTIFY = ("identify", "Confirm identity",
            "Make sure you are speaking with the owner or someone who can answer for the business. Wrong person: ask "
            "for the right person or a better time, then close politely.")
CLOSE = ("close", "Close",
         "Confirm what was captured or agreed in one sentence, say what happens next, thank them and say goodbye.")


# ---- 1. Seller enrichment ---------------------------------------------------------------------
ENRICH_FIELDS = {
    "product_range": "main products they sell", "brands": "brands they deal in",
    "service_locations": "cities / areas they supply to", "moq": "minimum order quantity, with unit",
    "manufacturing_capacity": "monthly production capacity, with unit", "business_type": "business type",
    "year_established": "year the business started", "employees": "number of employees",
    "delivery_time": "usual delivery time",
}


def _enrich_checklist(row: dict) -> list[tuple[str, str]]:
    keys = [_key(f) for f in _list(row.get("profile_missing_fields"))] or ["product_range", "service_locations"]
    return [(k, ENRICH_FIELDS.get(k, k.replace("_", " "))) for k in keys]


def _choose_enrich(t: dict, row: dict, stype: str) -> tuple[str, str]:
    nature = (t["nature"] or "").lower()
    if "manufactur" in nature or "exporter" in nature or t["high_turnover"]:
        return "manufacturer", f"{t['nature'] or 'Large business'}: can give detailed operational information"
    if any(w in nature for w in ("retail", "trader", "wholesal", "dealer", "shop", "distributor")) or stype == "time_pressed":
        return "shopkeeper", f"{t['nature'] or 'Busy trader'}: short of time, prefers a quick friendly chat"
    return "shopkeeper", "No sign of a large operation: friendly conversational default"


ENRICHMENT = UseCase(
    id="seller_enrichment", name="Seller enrichment",
    goal="Collect or verify the missing business details in the seller's IndiaMART profile.",
    metric="Share of calls that end with a complete, verified seller profile.",
    ask="the few missing profile details", benefit="a complete profile shows up for more relevant buyers",
    facts=_facts("Payal is updating the seller's IndiaMART business profile: a complete profile is shown to more "
                 "relevant buyers.", "The details are updated on their profile; nothing to pay, nothing to sign."),
    flow=_flow("seller_enrichment", "Seller enrichment",
               "You are Payal, IndiaMART's voice assistant, on a call with a seller to complete their IndiaMART business "
               "profile. Collect or verify only the missing details listed in the checklist, quickly and accurately.",
               [IDENTIFY,
                ("reason", "Reason for the call", "In one line: their profile is missing a few details; completing it "
                 "helps more relevant buyers find them. Ask if two minutes is fine."),
                ("collect", "Collect details", "Go through the checklist one item per question. Capture numbers with "
                 "units. If they give several details at once, capture all of them."),
                ("verify", "Verify", "Read back only numbers or spellings that are easy to get wrong (MOQ, capacity, "
                 "brand names) in one short sentence."),
                CLOSE],
               ("Why do you need this: so buyers searching for their products see accurate details and contact them.",
                "Will it cost anything: no, updating the profile is free.",
                "Already updated on the app: thank them; verify only what is still missing."),
               ("product_range", "brands", "service_locations", "moq", "manufacturing_capacity", "objection", "callback")),
    archetypes={
        "shopkeeper": Archetype("shopkeeper", "Busy shopkeeper", "normal",
                                "Friendly and conversational: weave the questions into a light chat, accept approximate "
                                "answers, and skip anything they find hard. Two minutes at most.",
                                {"sentence_length": "short", "vocabulary": "everyday", "warmth": "high"}),
        "manufacturer": Archetype("manufacturer", "Manufacturer / large business", "advisor",
                                  "Structured: tell them there are N quick points, go in a fixed order, and capture "
                                  "exact figures with units (capacity per month, MOQ). Professional and precise.",
                                  {"sentence_length": "normal", "vocabulary": "business"}),
    },
    choose=_choose_enrich,
    context=lambda row: [f"Missing from their profile: {', '.join(_list(row.get('profile_missing_fields'))) or 'unknown'}."],
    checklist=_enrich_checklist,
)


# ---- 2. Meeting scheduling (the original VANI flow) -------------------------------------------------
def _choose_meeting(t: dict, row: dict, stype: str) -> tuple[str, str]:
    if stype == "time_pressed":
        return "rushed", "Past calls were short or deferred"
    if t["new_seller"] or t["prior_meeting"] or (t["enquiries_90d"] or 0) >= 20:
        return "interested", ("New seller" if t["new_seller"] else "Met before" if t["prior_meeting"]
                              else "Getting buyer enquiries") + ": likely receptive"
    if stype == "guarded" or "not_ready" in t["objections"] or t["questions"]:
        return "explain", "Asked questions or hesitated on past calls: needs an explanation before agreeing"
    return "explain", "No strong signal: explain the value first, then ask"


MEETING = UseCase(
    id="meeting_scheduling", name="Meeting scheduling",
    goal="Schedule a short meeting with an IndiaMART sales executive / relationship manager.",
    metric="Meeting booking rate, confirmed attendance and fewer reschedules.",
    ask="a short meeting", benefit="the executive shows how to get more buyers",
    facts=MEETING_FACTS, flow=SELLER,
    archetypes={
        "rushed": Archetype("rushed", "Rushed seller", "quick_direct",
                            "Brief and direct: one line of purpose, then the time slot or a callback.",
                            {"sentence_length": "very_short"}),
        "interested": Archetype("interested", "Interested seller", "closer",
                                "Warm and persuasive: connect the meeting to their enquiries and close a concrete slot.",
                                {"warmth": "high"}),
        "explain": Archetype("explain", "Needs an explanation", "patient_explainer",
                             "Patient: explain what the meeting covers and why it helps, answer questions, then ask.",
                             {"vocabulary": "simple"}),
    },
    choose=_choose_meeting,
    context=lambda row: [],
    checklist=lambda row: [("meeting_day", "meeting day"), ("meeting_time", "meeting time"),
                           ("meeting_mode", "visit or online")],
)


# ---- 3. Lead qualification and follow-up ---------------------------------------------------------------
def _enquiry_lines(row: dict) -> list[str]:
    bits = []
    if _str(row.get("enquiry_product")):
        q = f", {row['enquiry_quantity']}" if _str(row.get("enquiry_quantity")) else ""
        city = f" from a buyer in {row['enquiry_buyer_city']}" if _str(row.get("enquiry_buyer_city")) else ""
        age = (f", received {int(_num(row['enquiry_age_hours']))} hours ago" if _num(row.get("enquiry_age_hours"))
               is not None else "")
        bits.append(f"Buyer enquiry for {row['enquiry_product']}{q}{city}{age}.")
    if _num(row.get("unreplied_enquiries")):
        bits.append(f"{int(_num(row['unreplied_enquiries']))} buyer enquiries in the last 30 days have no reply yet.")
    return bits


def _experienced(t: dict, row: dict) -> bool:
    replies = _num(row.get("eng_enq_replies_1yr")) or 0
    products = _num(row.get("eng_product_count")) or 0
    gst = _num(row.get("gst_registration_year"))
    return replies >= 50 or products >= 50 or (gst is not None and gst <= 2018)


def _choose_qualify(t: dict, row: dict, stype: str) -> tuple[str, str]:
    if _experienced(t, row):
        return "experienced", "Long on IndiaMART / replies to many enquiries: knows how leads work"
    return "unfamiliar", "New or low activity on IndiaMART: may not know how enquiries work"


QUALIFY = UseCase(
    id="lead_qualification", name="Lead qualification and follow-up",
    goal="Check whether a buyer enquiry is relevant, genuine and actionable for the seller, and whether they will respond.",
    metric="Qualified leads, seller response rate and follow-up completion.",
    ask="a quick check on this buyer enquiry", benefit="a genuine buyer is waiting for their reply",
    facts=_facts("Payal is following up on a buyer enquiry the seller received on IndiaMART.",
                 "If the seller wants it, the buyer's details stay in their IndiaMART enquiries so they can call or "
                 "quote; irrelevant enquiries are marked so they get better-matched ones.",
                 {"enquiry_rule": "Payal only knows the enquiry details in the call context; she never invents buyer names "
                                  "or numbers."}),
    flow=_flow("lead_qualification", "Lead qualification",
               "You are Payal, IndiaMART's voice assistant, following up with a seller about a buyer enquiry they "
               "received. Find out whether it is relevant and genuine for them and whether they will respond.",
               [IDENTIFY,
                ("enquiry", "Mention the enquiry", "Name the enquiry from the call context in one line: product, "
                 "quantity, buyer city."),
                ("relevance", "Relevance", "Do they deal in this product / can they supply this quantity to that city?"),
                ("genuineness", "Genuineness", "Have they already spoken to this buyer; does it look genuine to them?"),
                ("next_step", "Next step", "Will they call or quote the buyer, and by when? If not relevant, what "
                 "would be relevant for them?"),
                CLOSE],
               ("Who is the buyer / give the number: the buyer's details are in their IndiaMART enquiries (app or "
                "seller panel); Payal does not read out numbers.",
                "These leads are fake: ask what made it look fake, note it, offer better-matched enquiries."),
               ("relevant", "can_supply", "genuine", "will_respond", "respond_by", "reason", "callback")),
    archetypes={
        "experienced": Archetype("experienced", "Experienced seller", "normal",
                                 "Concise and transactional: product, quantity, city in one line, then 'aap supply kar "
                                 "paayenge? kab tak reply karenge?'. No explanation of how enquiries work.",
                                 {"sentence_length": "short", "vocabulary": "business"}),
        "unfamiliar": Archetype("unfamiliar", "Unfamiliar with the platform", "patient_explainer",
                                "Explanatory: say what a buyer enquiry is and where to find it in the IndiaMART app, then "
                                "ask the questions one at a time.",
                                {"vocabulary": "simple"}),
    },
    choose=_choose_qualify,
    context=_enquiry_lines,
    checklist=lambda row: [("relevant", "is the enquiry relevant"), ("genuine", "does it look genuine"),
                           ("will_respond", "will they respond"), ("respond_by", "when")],
)


# ---- 4. Buyer lead follow-up and response recovery ----------------------------------------------------------
def _choose_recover(t: dict, row: dict, stype: str) -> tuple[str, str]:
    if stype == "time_pressed" or t["prior_call_later"]:
        return "busy", "Short or deferred past calls: likely too busy to reply"
    if t["prior_not_interested"] or stype == "guarded":
        return "disengaged", "Said not interested before: find the blocker, no pressure"
    if t["new_seller"] or not _experienced(t, row):
        return "confused", "New / low activity: may not know where enquiries arrive"
    return "busy", "Experienced seller with unanswered enquiries: most often a time problem"


RECOVER = UseCase(
    id="response_recovery", name="Buyer lead follow-up and response recovery",
    goal="Find out why the seller has not replied to buyer enquiries and get an appropriate next step.",
    metric="Recovered responses and time to first seller response.",
    ask="a reply to the waiting buyers", benefit="buyers who get a quick reply usually buy from that seller",
    facts=_facts("Payal is calling because buyers sent the seller enquiries on IndiaMART that have no reply yet.",
                 "The seller can reply from the IndiaMART app (Enquiries tab) or call the buyer; Payal can call back "
                 "at a better time."),
    flow=_flow("response_recovery", "Response recovery",
               "You are Payal, IndiaMART's voice assistant. Buyers sent this seller enquiries that have no reply. "
               "Understand why, without blaming, and agree a realistic next step.",
               [IDENTIFY,
                ("enquiry", "Unanswered enquiries", "Mention the waiting enquiries from the call context in one line."),
                ("reason", "Why no reply", "Ask gently what stopped them: busy, did not see it, not relevant, price, "
                 "app trouble."),
                ("resolve", "Address the blocker", "Busy: offer a short callback. Did not see / confused: explain where "
                 "enquiries arrive. Not relevant: note what is. Never push."),
                ("next_step", "Next step", "Agree when they will reply or call the buyer, or a callback time."),
                CLOSE],
               ("Did not get any enquiry: say they are in the IndiaMART app, Enquiries tab, and in SMS/email alerts.",
                "Enquiries are not relevant: note which products they want enquiries for."),
               ("reason", "blocker", "will_respond", "respond_by", "callback", "relevant_products")),
    archetypes={
        "busy": Archetype("busy", "Busy seller", "quick_direct",
                          "Respect their time: one line about the waiting buyers, then offer a short callback or ask "
                          "when they can reply today.", {"sentence_length": "very_short"}),
        "confused": Archetype("confused", "Unsure where enquiries are", "patient_explainer",
                              "Explain the enquiry: who asked, for what, and where to see it in the app, one step at a "
                              "time.", {"vocabulary": "simple"}),
        "disengaged": Archetype("disengaged", "Uninterested seller", "gentle_nudger",
                                "Identify the blocker without being pushy: ask what is not working for them and accept "
                                "the answer.", {"empathy": "high"}),
    },
    choose=_choose_recover,
    context=_enquiry_lines,
    checklist=lambda row: [("reason", "why no reply"), ("blocker", "what blocks them"),
                           ("will_respond", "will they reply"), ("respond_by", "when / callback time")],
)


# ---- 5. Seller onboarding and activation -------------------------------------------------------------------
def _choose_onboard(t: dict, row: dict, stype: str) -> tuple[str, str]:
    products = _num(row.get("eng_product_count")) or 0
    modified = _num(row.get("eng_products_modified_90d")) or 0
    digital = _str(row.get("digital_experience"))
    if (digital or "").lower() in ("high", "experienced") or products >= 15 or modified >= 10:
        return "digital", "Already active online / lists products themselves"
    return "beginner", "Few products listed and little activity: first steps on IndiaMART"


ONBOARD = UseCase(
    id="onboarding", name="Seller onboarding and activation",
    goal="Help a newly registered seller finish onboarding and take the first step towards buyer enquiries.",
    metric="Onboarding completion and time to first meaningful platform activity.",
    ask="the next onboarding step", benefit="a complete catalogue starts bringing buyer enquiries",
    facts=_facts("Payal is helping the seller set up their new IndiaMART account so buyers can find them.",
                 "Each step is done in the IndiaMART seller app; Payal can call back to help with the next one."),
    flow=_flow("onboarding", "Onboarding",
               "You are Payal, IndiaMART's voice assistant, helping a newly registered seller complete their onboarding "
               "on IndiaMART and take their first step towards buyer enquiries.",
               [IDENTIFY,
                ("welcome", "Welcome", "Welcome them to IndiaMART in one line and say what is left to set up."),
                ("steps", "Pending steps", "Go through the pending steps from the call context in order; for each, "
                 "ask if it is done or help them do it now."),
                ("first_activity", "First activity", "Agree one concrete action today (e.g. add ten products with "
                 "photos and prices) and when."),
                CLOSE],
               ("How do I add products: IndiaMART seller app, 'Add Product': name, photo, price, short description.",
                "When will I get enquiries: once products are listed with photos and prices, buyers can find them."),
               ("done_steps", "blocked_on", "first_action", "action_by", "callback")),
    archetypes={
        "beginner": Archetype("beginner", "New to selling online", "patient_explainer",
                              "One step at a time: explain a step, check it is done, then the next. Simple words, no "
                              "jargon, praise progress.", {"vocabulary": "simple", "sentence_length": "short"}),
        "digital": Archetype("digital", "Digitally experienced", "advisor",
                             "Results-oriented: list what is pending, link each step to more enquiries, let them do it "
                             "themselves.", {"vocabulary": "business"}),
    },
    choose=_choose_onboard,
    context=lambda row: ([f"Pending onboarding steps: {', '.join(_list(row.get('onboarding_steps_pending')))}."]
                         if _list(row.get("onboarding_steps_pending")) else [])
    + ([f"Registered {int(_num(row['registered_days_ago']))} days ago."] if _num(row.get("registered_days_ago")) else []),
    checklist=lambda row: [(_key(s), s) for s in _list(row.get("onboarding_steps_pending"))]
    or [("first_action", "first action")],
)


# ---- 6. Subscription renewal and upgrade -------------------------------------------------------------------
def _choose_renew(t: dict, row: dict, stype: str) -> tuple[str, str]:
    sat = (_str(row.get("satisfaction")) or "").lower()
    if sat in ("low", "dissatisfied") or t["frustration"] or "bad_past_experience" in t["objections"]:
        return "dissatisfied", "Low satisfaction / complaints on record"
    if _num(row.get("renewed_before")) == 1 and sat in ("high", "satisfied"):
        return "informed", "Renewed before and satisfied: already understands the offering"
    return "evaluating", "Needs to judge the value before renewing"


RENEW = UseCase(
    id="renewal", name="Subscription renewal and upgrade",
    goal="Talk to an eligible seller about the upcoming renewal, their experience and plan benefits, and capture "
         "renewal / upgrade interest.",
    metric="Renewal conversion, qualified upgrade interest and fewer unresolved objections.",
    ask="their renewal decision", benefit="continued visibility and buyer enquiries without a break",
    facts=_facts("Payal is calling about the seller's IndiaMART subscription, which is up for renewal.",
                 "An IndiaMART account manager confirms renewal prices and plan details and completes the renewal.",
                 {"plan_rule": "Payal may mention the plan name and expiry from the call context, never prices or "
                               "discounts."}),
    flow=_flow("renewal", "Renewal",
               "You are Payal, IndiaMART's voice assistant, calling a paid seller whose subscription is up for renewal. "
               "Understand their experience, address concerns honestly, and capture renewal or upgrade interest.",
               [IDENTIFY,
                ("reminder", "Renewal reminder", "Mention the plan and expiry from the call context in one line."),
                ("experience", "Experience", "Ask how the plan has worked for them: enquiries, quality, any problems."),
                ("value", "Value and concerns", "Answer concerns from FACTS and their own data (enquiries received); "
                 "complaints are noted for the account manager."),
                ("decision", "Renewal / upgrade", "Ask if they want to renew, and whether they want more (another "
                 "city, more categories); the account manager follows up with prices."),
                CLOSE],
               ("What is the renewal price / discount: the account manager confirms it; Payal does not quote.",
                "Leads were poor: acknowledge, ask what was wrong (relevance, quantity, location) and note it."),
               ("experience", "concern", "renewal_intent", "upgrade_interest", "followup_time", "callback")),
    archetypes={
        "dissatisfied": Archetype("dissatisfied", "Dissatisfied seller", "empathetic_listener",
                                  "Empathetic: let them explain what went wrong before mentioning renewal; note the "
                                  "issue for the account manager; no pressure.", {"empathy": "high"}),
        "evaluating": Archetype("evaluating", "Evaluating the value", "advisor",
                                "Consultative: ask what they need from IndiaMART and relate it to what the plan "
                                "delivered for them (their enquiry numbers).", {"vocabulary": "business"}),
        "informed": Archetype("informed", "Knows the offering", "quick_direct",
                              "Concise: confirm the expiry, ask if they want to renew as is or upgrade, fix the account "
                              "manager follow-up.", {"sentence_length": "short"}),
    },
    choose=_choose_renew,
    context=lambda row: [line for line in [
        (f"Plan: {row['subscription_plan']}" + (f", expires in {int(_num(row['subscription_expiry_days']))} days"
                                                  if _num(row.get("subscription_expiry_days")) is not None else "") + ".")
        if _str(row.get("subscription_plan")) else None,
        (f"Buyer enquiries on this plan: {int(_num(row['enquiries_on_plan']))}." if _num(row.get("enquiries_on_plan"))
         is not None else None),
        f"Recorded satisfaction: {row['satisfaction']}." if _str(row.get("satisfaction")) else None,
    ] if line],
    checklist=lambda row: [("experience", "their experience"), ("renewal_intent", "renewal intent"),
                           ("upgrade_interest", "upgrade interest"), ("followup_time", "account-manager follow-up")],
)


# ---- 7. Seller feedback and issue resolution -----------------------------------------------------------------
def _choose_issue(t: dict, row: dict, stype: str) -> tuple[str, str]:
    age = _num(row.get("ticket_age_days")) or 0
    contacts = _num(row.get("ticket_prior_contacts")) or 0
    if t["frustration"] or age >= 7 or contacts >= 2:
        return "frustrated", (f"Issue open {int(age)} days" if age >= 7 else
                              f"Contacted {int(contacts)} times already" if contacts >= 2 else "Showed frustration before")
    if t["new_seller"]:
        return "confused", "New seller: may need the issue explained in simple steps"
    return "routine", "Recent issue, first follow-up"


ISSUE = UseCase(
    id="issue_resolution", name="Seller feedback and issue resolution",
    goal="Understand the seller's problem after a complaint or service issue, assess severity and route it to the "
         "right team.",
    metric="Issue classification accuracy, successful escalation and seller satisfaction.",
    ask="the details of the issue", benefit="the right team fixes it faster",
    facts=_facts("Payal is following up on the issue the seller reported to IndiaMART.",
                 "Payal records the issue and routes it to the right IndiaMART team (Support, Billing, Leads, Catalogue, "
                 "Technical) who contact the seller; Payal cannot fix it on the call or promise a resolution date.",
                 {"ticket_rule": "The issue details in the call context are already known: do not ask for them again."}),
    flow=_flow("issue_resolution", "Issue resolution",
               "You are Payal, IndiaMART's voice assistant, following up on an issue a seller reported. Listen, "
               "understand the problem and its impact, assess severity, and route it to the right team.",
               [IDENTIFY,
                ("acknowledge", "Acknowledge", "Refer to their issue from the call context in one line, so they do not "
                 "have to repeat it; apologise once if it is still open."),
                ("understand", "Understand", "Ask only what is still unclear: is it still happening, since when, what "
                 "exactly fails."),
                ("severity", "Severity", "Assess impact: are they losing business, how urgent is it."),
                ("route", "Route", "Say which team will handle it and that they will contact the seller."),
                CLOSE],
               ("When will it be fixed: the team will contact them; Payal cannot promise a date.",
                "I already explained this many times: apologise, do not ask again, confirm what you already know."),
               ("still_open", "details", "severity", "impact", "team", "preferred_contact_time")),
    archetypes={
        "frustrated": Archetype("frustrated", "Frustrated seller", "empathetic_listener",
                                "Empathy first: acknowledge the delay and that they have explained it before; never ask "
                                "for information already in the context.", {"empathy": "high", "warmth": "calm"}),
        "confused": Archetype("confused", "Confused seller", "patient_explainer",
                              "Slow down: explain what happens next in simple steps.", {"vocabulary": "simple"}),
        "routine": Archetype("routine", "Routine follow-up", "normal",
                             "Clear and efficient: confirm the issue, check severity, route it.",
                             {"sentence_length": "short"}),
    },
    choose=_choose_issue,
    context=lambda row: [line for line in [
        (f"Reported issue ({row.get('ticket_type') or 'general'}): {row['ticket_summary']}"
         + (f", open {int(_num(row['ticket_age_days']))} days" if _num(row.get("ticket_age_days")) is not None else "")
         + (f", status {row['ticket_status']}" if _str(row.get("ticket_status")) else "") + ".")
        if _str(row.get("ticket_summary")) else None,
        (f"They have already contacted IndiaMART {int(_num(row['ticket_prior_contacts']))} times about it."
         if _num(row.get("ticket_prior_contacts")) else None),
    ] if line],
    checklist=lambda row: [("still_open", "is it still happening"), ("severity", "severity"),
                           ("impact", "business impact"), ("team", "team it goes to")],
)


# ---- 8. Product and catalogue verification ---------------------------------------------------------------------
def _listings(row: dict) -> list[str]:
    return _list(row.get("listings_to_verify"))


def _choose_catalogue(t: dict, row: dict, stype: str) -> tuple[str, str]:
    size = _num(row.get("catalogue_size")) or _num(row.get("eng_product_count")) or 0
    if size >= 50:
        return "large", f"Large catalogue ({int(size)} products): needs a systematic checklist"
    return "busy", "Small catalogue / busy seller: short, specific questions"


CATALOGUE = UseCase(
    id="catalogue_verification", name="Product and catalogue verification",
    goal="Verify outdated listings: availability, price, brand, minimum order quantity and other details.",
    metric="Verified listings, correction rate and less outdated catalogue information.",
    ask="a quick check of a few listings", benefit="accurate listings bring buyers who can actually buy",
    facts=_facts("Payal is checking that a few of the seller's IndiaMART product listings are still correct.",
                 "Corrections are updated on their listings; nothing to pay.",
                 {"price_rule": "Payal may read out the listed price from the call context to verify it, and record a "
                                "corrected price the seller states; she never suggests a price."}),
    flow=_flow("catalogue_verification", "Catalogue verification",
               "You are Payal, IndiaMART's voice assistant, verifying a seller's outdated product listings on IndiaMART.",
               [IDENTIFY,
                ("reason", "Reason", "Say a few listings look outdated and you want to confirm them in two minutes."),
                ("verify", "Verify listings", "For each listing in the call context: still available? price, MOQ, "
                 "brand still correct? Capture corrections exactly."),
                ("other", "Anything to remove", "Ask if any listed product should be removed or added."),
                CLOSE],
               ("Why does it matter: outdated prices or stock bring buyers who cannot buy, and waste their time.",),
               ("verified", "corrections", "remove", "add", "callback")),
    archetypes={
        "busy": Archetype("busy", "Busy seller", "quick_direct",
                          "Short, specific questions: one listing per question, yes / no or the corrected value.",
                          {"sentence_length": "very_short"}),
        "large": Archetype("large", "Large catalogue", "advisor",
                           "Systematic checklist: say how many listings, go in order, for each confirm availability, "
                           "price, MOQ and brand; offer to send the rest to their catalogue manager.",
                           {"vocabulary": "business"}),
    },
    choose=_choose_catalogue,
    context=lambda row: ([f"Listings to verify ({len(_listings(row))}): " + "; ".join(_listings(row)) + "."]
                         if _listings(row) else [])
    + ([f"Catalogue size: {int(_num(row['catalogue_size']))} products."] if _num(row.get("catalogue_size")) else []),
    checklist=lambda row: [(f"listing_{i}", ls.split("|")[0].strip()) for i, ls in enumerate(_listings(row), 1)]
    or [("verified", "listings verified")],
)


# ---- 9. Win back a churned seller --------------------------------------------------------------------------
BURNED_WORDS = ("irrelevant", "fake", "poor", "bad", "service", "quality", "complaint", "wrong", "no response",
                "spam", "fraud")
COMPETITOR_WORDS = ("competitor", "other platform", "another platform", "tradeindia", "justdial", "own website",
                    "switched")
PRICE_WORDS = ("price", "cost", "expensive", "costly", "budget", "afford", "slow", "business down", "roi", "not worth")


def _choose_winback(t: dict, row: dict, stype: str) -> tuple[str, str]:
    why = (_str(row.get("churn_reason")) or "").lower()
    if any(w in why for w in BURNED_WORDS) or t["frustration"] or "bad_past_experience" in t["objections"]:
        return "burned", f"Left after a bad experience ({why or 'frustration on record'}): has to be heard first"
    if _str(row.get("competitor")) or any(w in why for w in COMPETITOR_WORDS):
        return "competitor", f"Moved to {_str(row.get('competitor')) or 'another platform'}: respect the choice, stay curious"
    if any(w in why for w in PRICE_WORDS):
        return "price", f"Left over cost / value ({why}): needs the numbers, not a pitch"
    enq = _num(row.get("enquiries_on_plan")) or 0
    return "drifted", (f"No clear reason recorded and {int(enq)} enquiries on the plan: drifted away, warm reminder"
                       if enq else "No clear reason recorded: drifted away, warm reminder")


def _winback_context(row: dict) -> list[str]:
    plan = _str(row.get("churned_plan"))
    months, days = _num(row.get("months_on_plan")), _num(row.get("churned_days_ago"))
    lines = [
        (f"Was on the {plan} plan" + (f" for {int(months)} months" if months else "")
         + (f"; it ended {int(days)} days ago" if days is not None else "") + ".") if plan else None,
        f"Reason recorded when they left: {row['churn_reason']}." if _str(row.get("churn_reason")) else None,
        (f"Buyer enquiries while on the plan: {int(_num(row['enquiries_on_plan']))}."
         if _num(row.get("enquiries_on_plan")) is not None else None),
        f"Now using: {row['competitor']}." if _str(row.get("competitor")) else None,
    ]
    return [line for line in lines if line]


WINBACK = UseCase(
    id="win_back", name="Win back a churned seller",
    goal="Reconnect with a seller who left their paid IndiaMART plan, understand honestly why they left, and find "
         "out what would bring them back.",
    metric="Churn reasons captured, sellers open to an account-manager callback, and win-back conversion.",
    ask="a few minutes to hear why they left", benefit="buyers are still searching IndiaMART for what they sell",
    facts=_facts("Payal is calling a seller who used to have a paid IndiaMART plan, to understand why they left and "
                 "whether anything would make it worth coming back.",
                 "If they are open to it, an IndiaMART account manager calls them with suitable plan options; their "
                 "free listing stays active either way.",
                 {"plan_rule": "Payal may mention the old plan name and their enquiry numbers from the call context, "
                               "never prices, discounts or offers.",
                  "tone_rule": "This is not a sales call: listening and the honest reason matter more than a yes.",
                  "apology_rule": "Apologise at most once in the whole call, and only after the seller has described "
                                  "the problem; if you already apologised (see the history), acknowledge without "
                                  "saying sorry again."}),
    flow=_flow("win_back", "Win-back",
               "You are Payal, IndiaMART's voice assistant, calling a seller who had a paid IndiaMART plan and left. "
               "Thank them for having been with IndiaMART, understand honestly why they left, and find out what would "
               "bring them back. Never sound desperate and never pressure.",
               [IDENTIFY,
                ("reconnect", "Reconnect", "Thank them for having been a paid IndiaMART member and mention the plan "
                 "from the call context in one line. No pitch and no apology yet."),
                ("reason", "Why they left", "If a reason is in the call context, confirm it in their words ('aapne "
                 "bataya tha leads relevant nahi the, sahi?') instead of asking cold; otherwise ask openly."),
                ("what_changed", "What would change it", "Ask what would have to be different for IndiaMART to be "
                 "worth it again. Relate it to their own numbers; note complaints for the account manager."),
                ("comeback", "Open to coming back", "Ask if they would take a call from an account manager with "
                 "options that fit; if no, accept it and say the free listing stays active. Always record their "
                 "answer as comeback_interest (yes / maybe / no), and the agreed time as followup_time."),
                CLOSE],
               ("Why should I come back / what is new: do not oversell; ask what they need and say the account manager "
                "can show what fits.",
                "Price was too high: acknowledge; the account manager can discuss plan options; Payal does not quote.",
                "Leads were fake or irrelevant: acknowledge, ask which kind, note it; lead preferences can be set by "
                "product and city.",
                "I use another platform now: respect it, never criticise competitors; ask what works better there."),
               ("churn_reason", "what_would_change", "comeback_interest", "followup_time", "competitor", "callback")),
    archetypes={
        "burned": Archetype("burned", "Left after a bad experience", "empathetic_listener",
                            "Hear them out first: let them describe what went wrong, acknowledge it in their own words, "
                            "apologise only once in the whole call, note it for the account manager. Only then ask "
                            "what would need to change.",
                            {"empathy": "high", "warmth": "calm"}),
        "price": Archetype("price", "Left over cost / value", "advisor",
                           "Consultative: talk about value with their own numbers (enquiries on the plan), ask what "
                           "return they need, and offer an account-manager call about options that fit their budget.",
                           {"vocabulary": "business"}),
        "competitor": Archetype("competitor", "Moved to another platform", "gentle_nudger",
                                "Respectful curiosity: never criticise the other platform; ask what works better there "
                                "and what IndiaMART could do differently; keep the door open, no pressure.",
                                {"sentence_length": "short"}),
        "drifted": Archetype("drifted", "Drifted away", "closer",
                             "Warm reconnect: remind them of the buyers they got on IndiaMART, ask why they stopped, "
                             "and invite them back with an account-manager call.", {"warmth": "high"}),
    },
    choose=_choose_winback,
    context=_winback_context,
    checklist=lambda row: [("churn_reason", "why they left"), ("what_would_change", "what would bring them back"),
                           ("comeback_interest", "open to coming back"),
                           ("followup_time", "account-manager callback time")],
)


USE_CASES: dict[str, UseCase] = {u.id: u for u in (ENRICHMENT, MEETING, QUALIFY, RECOVER, ONBOARD, RENEW, ISSUE,
                                                   CATALOGUE, WINBACK)}
USE_CASE_FLOWS: dict[str, Flow] = {u.flow.mode: u.flow for u in USE_CASES.values()}
# A note that settles another checklist item: (mode -> {note: (implied note, value)}).
NOTE_IMPLIES = {"win_back": {"followup_time": ("comeback_interest", "yes, agreed to an account-manager call")}}
ALIASES = {"seller": "meeting_scheduling", "meeting": "meeting_scheduling", "enrichment": "seller_enrichment",
           "qualification": "lead_qualification", "recovery": "response_recovery", "catalogue": "catalogue_verification",
           "issue": "issue_resolution", "renewal_upgrade": "renewal", "winback": "win_back", "churn": "win_back",
           "churned": "win_back", "win_back_seller": "win_back", "reactivation": "win_back"}


def resolve(name: str | None) -> str | None:
    if not name:
        return None
    k = _key(name)
    return k if k in USE_CASES else ALIASES.get(k)


def infer_use_case(row: dict) -> tuple[str, str]:
    """(use case id, reason): the explicit column wins, else the most urgent need in the data."""
    explicit = resolve(_str(row.get("use_case")))
    if explicit:
        return explicit, "Set in the seller file (use_case column)."
    expiry = _num(row.get("subscription_expiry_days"))
    rules = [
        (_str(row.get("ticket_summary")) and (_str(row.get("ticket_status")) or "open").lower() != "closed",
         "issue_resolution", "Open issue on record: follow up on it first."),
        (_str(row.get("churned_plan")) or _num(row.get("churned_days_ago")) is not None, "win_back",
         "Had a paid plan and left: win-back call."),
        (expiry is not None and expiry <= 60, "renewal", f"Subscription expires in {int(expiry or 0)} days."),
        ((_num(row.get("unreplied_enquiries")) or 0) > 0, "response_recovery",
         f"{int(_num(row.get('unreplied_enquiries')) or 0)} buyer enquiries without a reply."),
        (_str(row.get("enquiry_product")), "lead_qualification", "A recent buyer enquiry needs qualifying."),
        (_list(row.get("onboarding_steps_pending")), "onboarding", "Onboarding steps still pending."),
        (_list(row.get("listings_to_verify")), "catalogue_verification", "Listings flagged as outdated."),
        (_list(row.get("profile_missing_fields")), "seller_enrichment", "Profile fields missing."),
    ]
    for hit, uid, why in rules:
        if hit:
            return uid, "Inferred from the seller file: " + why
    return "meeting_scheduling", "No specific need in the seller file: default meeting-scheduling call."


GENERIC_PLAYBOOK = {
    "busy_now": "Acknowledge in one line; offer to finish in two minutes now or call back at a time they pick.",
    "call_later": "Ask for one specific callback time and confirm it back; do not restart the call.",
    "not_interested": "Accept once; give one reason this helps them ({benefit}); if still no, thank them and end.",
    "already_in_touch": "Acknowledge their IndiaMART executive; say this call only needs {ask} and keep it short.",
    "not_ready": "Normalise it; offer to do one small part now or call back later.",
    "trust": "State identity plainly (IndiaMART, AI assistant Payal), explain why you are calling, never pressure.",
    "bad_past_experience": "Apologise for the past experience once, keep it short, note it for the right team.",
    "not_relevant": "Check what they deal in; if truly irrelevant, note it, thank them and end.",
    "price": "Do not quote; the IndiaMART team confirms prices and plans.",
    "is_this_a_bot": "Confirm truthfully that you are IndiaMART's AI assistant, then continue in one short line.",
    "send_details": "Say the IndiaMART team can share details after the call; ask only the one question needed now.",
}


def apply_use_case(spec: dict, row: dict | None, use_case: str | None = None) -> dict:
    """Return a copy of a generated seller ``spec`` fitted to the use case (inferred if not given)."""
    row = row or {}
    if "voice" not in spec:          # do-not-call: no persona, no call
        return spec
    if use_case and resolve(use_case):
        uid, uc_reason = resolve(use_case), "Chosen for this call."
    else:
        uid, uc_reason = infer_use_case(row)
    uc = USE_CASES[uid]
    t = traits(row)
    arch_id, arch_reason = uc.choose(t, row, spec.get("seller_type", "standard"))
    arch = uc.archetypes[arch_id]
    out = copy.deepcopy(spec)
    out["call_mode"] = uc.flow.mode

    # Delivery tweaks of the archetype, each with its reason in the spec.
    why = f"{uc.name}, {arch.label.lower()}: {arch_reason}."
    d = arch.delivery
    for key, section in (("warmth", "tone"), ("empathy", "tone")):
        if key in d and out[section][key]["value"] != d[key]:
            out[section][key] = {**out[section][key], "value": d[key],
                                 "reason": f"{why} (was {out[section][key]['value']}: {out[section][key]['reason']})"}
    delivery = {k: {"value": d[k], "reason": why} for k in ("sentence_length", "vocabulary") if k in d}

    ctx = [line for line in uc.context(row) if line]
    checklist = uc.checklist(row)
    out["use_case"] = {
        "id": uid, "name": uc.name, "goal": uc.goal, "metric": uc.metric, "reason": uc_reason,
        "archetype": {"id": arch.id, "label": arch.label, "reason": arch_reason, "approach": arch.approach},
        "start_persona": arch.persona, "delivery": delivery,
        "context": ctx, "checklist": [{"key": k, "label": label} for k, label in checklist],
        "seller_name": _str(row.get("seller_name")), "company": _str(row.get("company_name")),
    }
    who = " / ".join(x for x in (_str(row.get("company_name")), _str(row.get("seller_name"))) if x)
    out["summary"] = f"{uc.name} · {arch.label}" + (f" · {who}" if who else "") + f" — {spec['summary']}"
    if uid == "meeting_scheduling":
        return out

    out["facts"] = uc.facts
    plan = out["conversation_plan"]
    plan["opening"] = {
        **plan["opening"],
        "steps": [s for s in plan["opening"]["steps"] if "meeting" not in s.lower() and "callback last time" not in s]
        + [f"Give the reason for the call in one line: {uc.goal}"],
    }
    plan["personalisation"] = [h for h in plan["personalisation"] if "visit" not in h["value"].lower()]
    for p in plan["objection_playbook"]:
        p["approach"] = GENERIC_PLAYBOOK.get(p["objection"], p["approach"]).format(ask=uc.ask, benefit=uc.benefit)
    plan["escalation_rules"] = [
        {**e, "when": e["when"].replace("refuses the meeting", f"refuses {uc.ask}"),
         "do": e["do"].replace("the executive will cover it in the meeting", "the IndiaMART team confirms prices")
                      .replace("Offer an executive callback within 11 am - 5 pm",
                               "Offer a callback from the right IndiaMART team")}
        for e in plan["escalation_rules"]
    ]
    return out
