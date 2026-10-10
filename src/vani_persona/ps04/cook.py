"""Write ``samples/cooked_sellers.csv``: synthetic sellers for every VANI use case.

    python -m vani_persona.ps04.cook

All names, numbers and businesses are invented. Columns are the PS07 seller-file columns
(same names and value formats) plus use-case columns:

- ``use_case``: seller_enrichment | meeting_scheduling | lead_qualification |
  response_recovery | onboarding | renewal | issue_resolution | catalogue_verification.
  Leave it blank to let the system infer it from the columns below.
- ``seller_name``, ``company_name``: who VANI is calling.
- enrichment: ``profile_missing_fields`` (``;``-separated: product_range, brands,
  service_locations, moq, manufacturing_capacity, delivery_time, ...).
- lead qualification / response recovery: ``enquiry_product``, ``enquiry_quantity``,
  ``enquiry_buyer_city``, ``enquiry_age_hours``, ``unreplied_enquiries``.
- onboarding: ``onboarding_steps_pending`` (``;``-separated), ``registered_days_ago``,
  ``digital_experience`` (low | high).
- renewal: ``subscription_plan``, ``subscription_expiry_days``, ``enquiries_on_plan``,
  ``satisfaction`` (low | medium | high), ``renewed_before`` (0/1).
- issue resolution: ``ticket_type``, ``ticket_summary``, ``ticket_age_days``,
  ``ticket_status``, ``ticket_prior_contacts``.
- catalogue verification: ``listings_to_verify`` (``;`` between listings, ``|`` between
  fields), ``catalogue_size``.
"""

from __future__ import annotations

import pandas as pd

from vani_persona import REPO_ROOT

OUT = REPO_ROOT / "samples" / "cooked_sellers.csv"

BASE = {
    "seller_district": None, "seller_pincode": None, "seller_locality": None, "nature_of_business_secondary": None,
    "top_category_2": None, "top_category_3": None, "top_parent_category": None, "num_categories": 4,
    "eng_activity_30d": 10, "eng_answered_calls_90d": 2, "eng_call_attempts_90d": 4, "eng_call_attempts_1yr": 12,
    "eng_pickup_ratio_90d": 0.5, "eng_pickup_ratio_1yr": 0.5, "eng_long_calls_90d": 1, "eng_short_calls_90d": 1,
    "eng_callback_90d": 0, "eng_callback_1yr": 0, "eng_meetings_1yr": 0, "eng_hot_meetings_1yr": 0,
    "eng_enq_received_90d": 8, "eng_enq_received_1yr": 30, "eng_enq_replies_90d": 4, "eng_enq_replies_1yr": 15,
    "eng_pns_received_90d": 3, "eng_pns_received_1yr": 10, "eng_pns_answered_90d": 2, "eng_pns_answered_1yr": 7,
    "eng_ni_count_1yr": 0, "eng_np_count_1yr": 0, "eng_product_count": 20, "eng_products_modified_90d": 2,
    "eng_products_modified_1yr": 6, "eng_cqs": 55, "bot_attempts": 3, "bot_answered": 2, "calls_meeting_fixed": 0,
    "calls_not_interested": 0, "calls_general": 1, "calls_call_later": 0, "bot_answer_rate": 0.67,
    "avg_answered_call_sec": 45, "past_objections": None, "past_questions_asked": None,
    "asked_if_talking_to_bot": 0, "showed_frustration": 0, "past_dispositions_detailed": None,
    "already_in_touch_with_executive": 0, "do_not_call_requested": 0,
}

SELLERS = [
    # 1. Seller enrichment: busy shopkeeper vs manufacturer.
    dict(fk_glusr_usr_id=9100001, use_case="seller_enrichment", seller_name="Ramesh Sharma",
         company_name="Sharma Bartan Bhandar", seller_city="Lucknow", seller_state="Uttar Pradesh",
         business_type="Proprietorship", annual_turnover="0 - 40 L", nature_of_business="Trader - Retailer",
         gst_registration_year=2019, top_category_1="Steel Kitchen Utensils", top_category_group="Kitchen Utensils",
         avg_answered_call_sec=19, calls_call_later=2, past_objections="Seller Unavailability & Scheduling Constraints (2)",
         profile_missing_fields="brands; service_locations; delivery_time"),
    dict(fk_glusr_usr_id=9100002, use_case="seller_enrichment", seller_name="Kiran Patel",
         company_name="Patel Polymers Pvt Ltd", seller_city="Ahmedabad", seller_state="Gujarat",
         business_type="Limited Company", annual_turnover="5 - 25 Cr", nature_of_business="Manufacturer",
         gst_registration_year=2012, top_category_1="HDPE Pipes", top_category_group="Plastic Pipes & Fittings",
         eng_product_count=64, avg_answered_call_sec=95,
         profile_missing_fields="manufacturing_capacity; moq; brands; service_locations"),
    # 2. Meeting scheduling: rushed vs interested.
    dict(fk_glusr_usr_id=9100003, use_case="meeting_scheduling", seller_name="Sunil Verma",
         company_name="Verma Electricals", seller_city="Delhi", seller_state="Delhi", business_type="Proprietorship",
         annual_turnover="40 L - 1.5 Cr", nature_of_business="Trader - Wholesaler/Distributor",
         gst_registration_year=2016, top_category_1="Electrical Wires", top_category_group="Electrical Equipment and Supplies",
         avg_answered_call_sec=14, calls_call_later=3, bot_attempts=6, bot_answered=4,
         past_objections="Seller Unavailability & Scheduling Constraints (3)"),
    dict(fk_glusr_usr_id=9100004, use_case="meeting_scheduling", seller_name="Ananya Rao",
         company_name="Green Leaf Organics", seller_city="Bengaluru", seller_state="Karnataka",
         business_type="Partnership", annual_turnover="0 - 40 L", nature_of_business="Manufacturer",
         gst_registration_year=2025, top_category_1="Organic Fertilizer", top_category_group="Agriculture",
         eng_enq_received_90d=32, avg_answered_call_sec=70, past_questions_asked="Benefit (1); Meeting Visit (1)"),
    # 3. Lead qualification: experienced vs unfamiliar.
    dict(fk_glusr_usr_id=9100005, use_case=None, seller_name="Mahesh Agarwal", company_name="Shree Ganesh Steel",
         seller_city="Raipur", seller_state="Chhattisgarh", business_type="Partnership", annual_turnover="1.5 - 5 Cr",
         nature_of_business="Trader - Wholesaler/Distributor", gst_registration_year=2011,
         top_category_1="TMT Bars", top_category_group="Building Construction Material", eng_enq_replies_1yr=240,
         eng_product_count=85, enquiry_product="MS TMT bars 12 mm", enquiry_quantity="20 tonnes",
         enquiry_buyer_city="Bilaspur", enquiry_age_hours=6),
    dict(fk_glusr_usr_id=9100006, use_case="lead_qualification", seller_name="Lakshmi Iyer",
         company_name="Kaveri Handlooms", seller_city="Salem", seller_state="Tamil Nadu", business_type="Proprietorship",
         annual_turnover="0 - 40 L", nature_of_business="Manufacturer", gst_registration_year=2025,
         top_category_1="Cotton Bedsheets", top_category_group="Home Textiles", eng_enq_replies_1yr=3,
         eng_product_count=6, enquiry_product="cotton bedsheets, double size", enquiry_quantity="200 pieces",
         enquiry_buyer_city="Chennai", enquiry_age_hours=20),
    # 4. Response recovery: busy vs uninterested.
    dict(fk_glusr_usr_id=9100007, use_case=None, seller_name="Harpreet Singh", company_name="Singh Auto Parts",
         seller_city="Ludhiana", seller_state="Punjab", business_type="Proprietorship", annual_turnover="40 L - 1.5 Cr",
         nature_of_business="Trader - Retailer", gst_registration_year=2015, top_category_1="Tractor Spare Parts",
         top_category_group="Automobiles, Spare Parts and Accessories", avg_answered_call_sec=21, calls_call_later=2,
         unreplied_enquiries=7, enquiry_product="tractor clutch plates"),
    dict(fk_glusr_usr_id=9100008, use_case="response_recovery", seller_name="Deepak Jain",
         company_name="Bharat Packaging", seller_city="Indore", seller_state="Madhya Pradesh",
         business_type="Proprietorship", annual_turnover="0 - 40 L", nature_of_business="Manufacturer",
         gst_registration_year=2018, top_category_1="Corrugated Boxes", top_category_group="Packaging Material",
         calls_not_interested=3, eng_ni_count_1yr=3, past_objections="Explicit Disinterest / Refusal (3)",
         unreplied_enquiries=12, enquiry_product="5 ply corrugated boxes"),
    # 5. Onboarding: beginner vs digitally experienced.
    dict(fk_glusr_usr_id=9100009, use_case="onboarding", seller_name="Rekha Devi", company_name="Maa Durga Sweets",
         seller_city="Patna", seller_state="Bihar", business_type="Proprietorship", annual_turnover="0 - 40 L",
         nature_of_business="Manufacturer", gst_registration_year=2026, top_category_1="Indian Sweets",
         top_category_group="Food Products", eng_product_count=1, eng_products_modified_90d=1, eng_cqs=12,
         bot_attempts=0, bot_answered=0, avg_answered_call_sec=None, registered_days_ago=5, digital_experience="low",
         onboarding_steps_pending="add products with photos and prices; add business address; verify GST number; "
                                  "turn on enquiry alerts in the app"),
    dict(fk_glusr_usr_id=9100010, use_case=None, seller_name="Rohan Mehta", company_name="UrbanThread Apparel",
         seller_city="Mumbai", seller_state="Maharashtra", business_type="Partnership", annual_turnover="40 L - 1.5 Cr",
         nature_of_business="Manufacturer", gst_registration_year=2025, top_category_1="Men's Shirts",
         top_category_group="Apparel, Clothing & Garments", eng_product_count=28, eng_products_modified_90d=25,
         eng_cqs=72, bot_attempts=0, bot_answered=0, avg_answered_call_sec=None, registered_days_ago=18,
         digital_experience="high",
         onboarding_steps_pending="add prices to all products; add a catalogue video; set lead preferences"),
    # 6. Renewal: dissatisfied vs already informed.
    dict(fk_glusr_usr_id=9100011, use_case=None, seller_name="Anil Gupta", company_name="Gupta Hardware & Tools",
         seller_city="Jaipur", seller_state="Rajasthan", business_type="Proprietorship", annual_turnover="1.5 - 5 Cr",
         nature_of_business="Trader - Wholesaler/Distributor", gst_registration_year=2010,
         top_category_1="Hand Tools", top_category_group="Hardware", showed_frustration=1,
         past_objections="Call Handling / Executive Gaps (2)", subscription_plan="Leader, 1 year",
         subscription_expiry_days=21, enquiries_on_plan=46, satisfaction="low", renewed_before=0),
    dict(fk_glusr_usr_id=9100012, use_case="renewal", seller_name="Vikram Kulkarni",
         company_name="Precision CNC Works Ltd", seller_city="Pune", seller_state="Maharashtra",
         business_type="Limited Company", annual_turnover="25 - 100 Cr", nature_of_business="Manufacturer",
         gst_registration_year=2008, top_category_1="CNC Machined Components", top_category_group="Industrial Machinery",
         eng_product_count=120, subscription_plan="Star, 1 year", subscription_expiry_days=30, enquiries_on_plan=310,
         satisfaction="high", renewed_before=1),
    # 7. Issue resolution: frustrated vs confused new seller.
    dict(fk_glusr_usr_id=9100013, use_case=None, seller_name="Imran Khan", company_name="Royal Furniture House",
         seller_city="Hyderabad", seller_state="Telangana", business_type="Proprietorship",
         annual_turnover="40 L - 1.5 Cr", nature_of_business="Manufacturer", gst_registration_year=2014,
         top_category_1="Wooden Sofa Sets", top_category_group="Furniture", showed_frustration=1,
         ticket_type="Leads", ticket_summary="getting enquiries for office chairs, which they do not make, and from "
                                             "buyers in other states", ticket_age_days=12, ticket_status="open",
         ticket_prior_contacts=3),
    dict(fk_glusr_usr_id=9100014, use_case="issue_resolution", seller_name="Pooja Reddy",
         company_name="Sai Pharma Distributors", seller_city="Vijayawada", seller_state="Andhra Pradesh",
         business_type="Proprietorship", annual_turnover="0 - 40 L", nature_of_business="Trader - Wholesaler/Distributor",
         gst_registration_year=2025, top_category_1="Surgical Gloves", top_category_group="Medical Supplies",
         ticket_type="Technical", ticket_summary="cannot log in to the seller app after changing her mobile number",
         ticket_age_days=2, ticket_status="open", ticket_prior_contacts=0),
    # 8. Catalogue verification: busy small catalogue vs large catalogue.
    dict(fk_glusr_usr_id=9100015, use_case=None, seller_name="Prakash Patil", company_name="Om Sai Krishi Kendra",
         seller_city="Nashik", seller_state="Maharashtra", business_type="Proprietorship", annual_turnover="0 - 40 L",
         nature_of_business="Trader - Retailer", gst_registration_year=2017, top_category_1="Drip Irrigation",
         top_category_group="Agriculture", avg_answered_call_sec=17, calls_call_later=1, catalogue_size=9,
         listings_to_verify="Drip irrigation pipe 16 mm | 12 rupees per metre | MOQ 500 metres; "
                            "Sprinkler set | 2450 rupees per set | brand Jain; HDPE water tank 1000 litre | 6800 rupees"),
    dict(fk_glusr_usr_id=9100016, use_case="catalogue_verification", seller_name="Subhash Das",
         company_name="Bengal Jute Exports Ltd", seller_city="Kolkata", seller_state="West Bengal",
         business_type="Limited Company", annual_turnover="5 - 25 Cr", nature_of_business="Manufacturer",
         gst_registration_year=2009, top_category_1="Jute Bags", top_category_group="Bags", eng_product_count=140,
         catalogue_size=140,
         listings_to_verify="Jute shopping bag | 45 rupees per piece | MOQ 1000 pieces; "
                            "Jute hessian cloth | 70 rupees per metre | MOQ 500 metres; "
                            "Jute yarn | 95 rupees per kg | MOQ 1 tonne; Jute wine bottle bag | 60 rupees | MOQ 500"),
    # Asked not to be called: suppressed, no persona.
    dict(fk_glusr_usr_id=9100017, use_case="meeting_scheduling", seller_name="Rakesh Kumar",
         company_name="Kumar Sanitary Store", seller_city="Kanpur", seller_state="Uttar Pradesh",
         business_type="Proprietorship", annual_turnover="0 - 40 L", nature_of_business="Trader - Retailer",
         gst_registration_year=2013, top_category_1="Bathroom Fittings", top_category_group="Sanitaryware",
         do_not_call_requested=1),
]

USE_CASE_COLS = [
    "use_case", "seller_name", "company_name", "profile_missing_fields", "enquiry_product", "enquiry_quantity",
    "enquiry_buyer_city", "enquiry_age_hours", "unreplied_enquiries", "onboarding_steps_pending",
    "registered_days_ago", "digital_experience", "subscription_plan", "subscription_expiry_days", "enquiries_on_plan",
    "satisfaction", "renewed_before", "ticket_type", "ticket_summary", "ticket_age_days", "ticket_status",
    "ticket_prior_contacts", "listings_to_verify", "catalogue_size",
]
PROFILE_COLS = [
    "fk_glusr_usr_id", "seller_city", "seller_state", "business_type", "annual_turnover", "nature_of_business",
    "gst_registration_year", "top_category_1", "top_category_group",
]


def main() -> None:
    rows = [{**BASE, **s} for s in SELLERS]
    for r in rows:
        if not r["bot_answered"]:
            r["bot_answer_rate"] = None
    df = pd.DataFrame(rows)
    rest = [c for c in df.columns if c not in PROFILE_COLS + USE_CASE_COLS]
    df = df.reindex(columns=PROFILE_COLS + USE_CASE_COLS + rest)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} sellers -> {OUT}")


if __name__ == "__main__":
    main()
