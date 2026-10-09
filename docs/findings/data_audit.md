# Data audit (generated)

Source: `` (loader: `ps07`); calls: 12354.

## Coverage

- Date range: 2026-07-08 11:50:57+05:30 → 2026-09-01 14:55:33+05:30
- Distinct sellers (pseudonymized): 11298
- Calls with turn timestamps: 100%
- Calls with ASR confidence: 0%
- Calls by month: {'2026-07': 5229, '2026-08': 6966, '2026-09': 159}
- Calls with any unknown-speaker turn: 1%
- PII redactions applied: 255
- Seller turns per call: median 4, p10 1, p90 10; zero seller turns: 4%

## Outcomes

- `objective_achieved` known: 100%; rate among known: 22.1%
- `downstream_conversion` known: 0%
- Disposition classes:
  - objective_completed: 25.1%
  - callback_requested: 22.1%
  - not_interested: 18.6%
  - seller_hung_up: 13.4%
  - other: 9.8%
  - no_response: 6.7%
  - wrong_party: 3.3%
  - opt_out: 1.1%
- Raw dispositions not mapped (edit `disposition_map`): ['Call Later / Busy|already_in_touch_with_im|already_met_exec', 'Call Later / Busy|already_in_touch_with_im|in_touch_with_exec', 'Call Later / Busy|info_requested|none', 'Call Later / Busy|other|none', 'General (talked)|-|-', 'General (talked)|already_in_touch_with_im|already_met_exec', 'General (talked)|already_in_touch_with_im|in_touch_with_exec', 'General (talked)|info_requested|none', 'General (talked)|other|none', 'Meeting Fixed|already_in_touch_with_im|already_met_exec', 'Meeting Fixed|already_in_touch_with_im|in_touch_with_exec', 'Meeting Fixed|info_requested|none', 'Meeting Fixed|other|none', 'Not Interested|already_in_touch_with_im|already_met_exec', 'Not Interested|already_in_touch_with_im|in_touch_with_exec', 'Not Interested|info_requested|none', 'Not Interested|other|none']

## Context distribution (confounders)

- `campaign_id`: main_vani 75%, arrowhead 16%, last_met_d 8%, main 1%
- `call_objective`: fix_meeting 100%
- `lead_type`: PIM 32%, PUA 31%, UA 26%, OLP 4%, PAM 3%, NUR 1%, SCHD 1%, PUT 1%
- `language_hint`: __unknown__ 100%
- `seller_category`: Building Construction Material, Equipment, Civil Engineering and Real Estate 11%, Apparel, Clothing & Garments 8%, Vegetables, Fruits, Grains, Dairy Products & Other FMCG and Grocery Items 6%, Consumer Electronics &  Household Appliances 5%, Housewares, Home Appliances, Household Decorations & Consumables 4%, Packaging Material, Supplies & Machines 4%, Electrical Equipment and Supplies 4%, Cosmetics, Toiletries & Personal Care Products 3%
- `region`: Gujarat 21%, Uttar Pradesh 18%, Maharashtra 16%, Delhi 11%, Rajasthan 8%, West Bengal 5%, Haryana 5%, Madhya Pradesh 4%
- `agent_version`: main_vani 75%, arrowhead 16%, last_met_d 8%, main 1%

## Feature availability

| feature | role | available in | median |
|---|---|---|---|
| `n_seller_turns` | context | 100% | 4 |
| `seller_mean_tokens` | segment | 96% | 5.5 |
| `short_reply_ratio` | segment | 96% | 0.5 |
| `long_reply_ratio` | segment | 96% | 0 |
| `question_rate` | segment | 96% | 0 |
| `clarification_rate` | segment | 96% | 0 |
| `brevity_request_rate` | segment | 96% | 0 |
| `time_pressure_rate` | segment | 96% | 0 |
| `objection_price_rate` | segment | 96% | 0 |
| `objection_trust_rate` | segment | 96% | 0 |
| `objection_not_interested_rate` | segment | 96% | 0 |
| `objection_already_have_rate` | segment | 96% | 0 |
| `info_request_rate` | segment | 96% | 0 |
| `affirmation_rate` | segment | 96% | 0.375 |
| `engagement_slope` | segment | 73% | 0.583 |
| `response_latency_median_ms` | segment | 95% | 85 |
| `overlap_rate` | segment | 95% | 0 |
| `devanagari_ratio` | language | 96% | 0 |
| `hindi_function_ratio` | language | 91% | 1 |
| `code_switch_rate` | language | 96% | 0 |
| `duration_s` | context | 100% | 54 |

## Decisions this audit drives

- Timestamp coverage decides whether `response_latency_median_ms` and `overlap_rate` can be persona features.
- Outcome coverage decides whether `outcome_association.md` can be produced and which A/B primary metric is measurable.
- Calls with zero or one seller turn cannot be segmented. If they are common, the early-exit problem is an opening problem rather than a persona problem.
