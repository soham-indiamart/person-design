# 09 · A/B testing against the current VANI

*Deliverable 10, plus brief §8.* Code: `src/vani_persona/experiment/`.

## 1. Groups

| Arm | Behaviour |
|---|---|
| **Control** | Today's VANI. `CallSession(mode="control")` composes with `neutral_default`, which must be verified to be identical to the production prompt/TTS. Explicit **language** requests are honoured in both arms, because VANI is assumed to honour them today. If it does not, language handling becomes part of the treatment |
| **Treatment** | Validated persona library + live matching + gated switching + explicit overrides (`mode="treatment"`) |
| **Shadow (pre-phase)** | Same traffic as control, with the matcher and switcher running and logging but nothing applied (`mode="shadow"`). Used to measure latency, matching distributions and switching behaviour at zero risk |

Held constant across arms:
- the call objective;
- `ObjectivePolicy` (disclosures, approved answers, allowed actions);
- opt-out handling;
- ASR, TTS voice and LLM model;
- calling schedules.

The only difference is the STYLE layer and the TTS pace parameters.

**Randomization.** By **seller** (`assign_arm(seller_key, experiment_id, {"control": 0.5, "treatment": 0.5})`). A seller is in the same arm on every call, which prevents within-seller contamination across repeated calls. Arms are independent across experiments (separate hash salt per `experiment_id`).

**Contamination checks:**
- sellers sharing a phone number (one number → several seller IDs): assign by the canonical contact;
- calls transferred between campaigns: they keep their arm;
- control calls showing persona events: alert, they must be zero;
- SRM, below.

**Stratification.** Report by campaign, objective and language. If arm imbalance within a large campaign is a concern, assign separately per campaign using the same hash with campaign in the salt.

**Like-for-like heterogeneity.** Run the matcher in shadow mode inside the **control** arm too. Each control call then gets a *predicted* persona, so "treatment vs control among sellers predicted as persona P" compares like with like. Without it, persona-level effects cannot be estimated without bias.

## 2. Metric dictionary

Unit of randomization: seller. Unit of measurement: call, unless noted. Analysis: seller-cluster bootstrap. Window: calls during the experiment; outcomes observed within the stated window after each call.

| Metric | Type | Definition | Numerator / denominator | Window |
|---|---|---|---|---|
| **Meeting-fixed rate** (objective completion) | **Primary** | `meeting_fixed = 1` and not flagged by the `incorrect_mf` evaluator | valid meetings / answered calls with ≥ 1 seller turn | Evaluator output ≤ 2 days |
| Incorrect-meeting rate | **Guardrail** | Meetings flagged by `incorrect_mf` (seller did not really agree) | flagged / meetings fixed | ≤ 2 days |
| Meeting held | Secondary (downstream) | The executive actually met the seller | held / meetings fixed | ≤ 14 days (feed to be built) |
| Downstream conversion | Secondary | Objective-specific business action (e.g. photos added) | sellers with action / sellers called | 14–30 days |
| Premature termination | Secondary (quality) | Seller hangs up before the objective's last step, excluding opt-outs and technical failures | such calls / connected calls | in call |
| Clarification / repeat request rate | Secondary (quality) | Seller turns tagged `clarification` | tagged turns / seller turns | in call |
| Objection resolution rate | Secondary (quality) | Objections raised that end the call resolved (in `resolved_objections` or followed by a completed step) | resolved / raised | in call |
| Brevity-request rate | Secondary (quality) | Seller asks VANI to be brief | calls with ≥ 1 / connected calls | in call |
| Interruption (overlap) rate | Secondary (quality) | `overlap_rate` (if timestamps) | seller turns overlapping / seller turns | in call |
| Dialogue completion | Secondary | All objective steps completed | calls / connected calls | in call |
| **Opt-out rate** | **Guardrail** | Opt-out detected or DND disposition | calls / connected calls | in call |
| **Complaint rate** | **Guardrail** | Complaints linked to the call | complaints / calls | 14 days |
| **p95 response latency** | **Guardrail** | End-of-seller-speech → start-of-TTS | per call p95, averaged | in call |
| Fallback / error rate | Guardrail | `matcher_fallback` events, persona runtime exceptions | events / treatment calls | in call |
| Cost per call | Guardrail | LLM tokens (incl. STYLE block) + infra | currency / call | in call |
| Persona matching accuracy | System | Agreement with blind human review | agreeing / reviewed | monthly sample |
| Per-persona precision / recall | System | As above, per persona | — | monthly sample |
| Turns to initial assignment | System | Seller turns until the first commit | median over committed calls | in call |
| Switch frequency / reversal rate | System | Switches per call; switches back to the previous persona | — | in call |
| No-confident-match rate | System | Calls never leaving neutral | calls / treatment calls | in call |

Longer calls are **not** a success measure. Duration is reported only as context and as a cost driver.

## 3. Statistical plan

1. **Baseline period.** Phase 1, ≥ 4 weeks of control-equivalent traffic. Measures the primary metric's rate, its variance, calls per seller (m) and the intra-seller correlation (ICC).
   - What PS07 already tells us: 11.5% of all 269,455 answered bot calls fixed a meeting (Apr–Sep 2026), with **2.07 calls per seller and an intra-seller correlation of 0.215, so the design effect is ≈ 1.23**. The ≈ 1.09 calls/seller of the transcript subset is not representative, and neither is its 22.1% meeting rate (selection). Sizing on the real baseline is in `docs/14-dataset-analysis-and-requirements.md` §2.1, including the dilution caveat: only ~38% of calls ever leave neutral.
   - Restrict the experiment to one bot version (`main_vani`, 75% of traffic) or stratify by it.
2. **MDE.** The smallest absolute lift worth shipping, agreed with the business before the test. Example of the form: "+2 points on objective completion".
3. **α = 0.05 two-sided, power = 0.8** (`config/thresholds.yaml: experiment`).
4. **Sample size.**
   ```bash
   .venv/bin/python -m vani_persona.experiment.power --baseline <measured rate> --mde <agreed lift> \
       --calls-per-seller <measured m> --icc <measured ICC> --daily-calls <eligible calls/day>
   ```
   This gives calls per arm × design effect `1 + (m−1)·ICC`, and a duration rounded up to whole weeks.

   *Illustrative arithmetic only (not a plan: no baseline has been measured).* Calls per arm before the design effect, from `n_per_arm`:

   | baseline \ MDE | +1 pt | +2 pts | +3 pts |
   |---|---|---|---|
   | 10% | 14,751 | 3,841 | 1,774 |
   | 20% | 25,583 | 6,510 | 2,943 |
   | 30% | 33,275 | 8,394 | 3,763 |

   Small lifts need large samples. That is why the MDE must be agreed up front, and why shadow mode should go first.
5. **Duration and stopping.**
   - Fixed horizon: the computed sample, in whole weeks, at least 2 full weeks.
   - **No early stopping for benefit.** Early looks are only for harm:
     - SRM check daily: `srm_pvalue < 0.001` → stop and debug assignment/logging;
     - guardrail breach → stop the treatment.
   - If interim efficacy looks are required, pre-register an alpha-spending plan (O'Brien–Fleming).
6. **Inference.**
   - `cluster_bootstrap_effect` resamples sellers within each arm, giving the difference in rates with a 95% CI and the relative lift.
   - Report the CI and **practical significance** (CI lower bound vs MDE), not just p < 0.05.
7. **Segments.** Pre-registered:
   - campaign, objective, language;
   - shadow-predicted persona (both arms);
   - new vs repeat seller.

   Segment results are exploratory unless pre-registered with a multiplicity correction (Holm).
8. **Guardrails** (`check_guardrails`). Non-inferiority: the worst plausible degradation (CI bound) must be within the allowed margin. Proposed margins, to be agreed:
   - opt-out ≤ +0.3 pts;
   - complaints ≤ +0.1 pts;
   - p95 latency ≤ +50 ms;
   - fallback rate ≤ 1%.

## 4. Rollout decisions

| Decision | Evidence required |
|---|---|
| **Reject** the persona system | At horizon: primary CI excludes the MDE and the point estimate ≤ 0, or any guardrail breached with the CI excluding the margin |
| **Continue** (extend once) | CI includes both 0 and the MDE, guardrails clean, and the pre-registered max duration not reached |
| **Limited rollout** (e.g. 50% → treatment default for the tested campaigns) | Primary CI lower bound > 0 **and** point estimate ≥ MDE; all guardrails clean; no SRM; no pre-registered segment with a significant negative effect |
| **Expand** to new campaigns / objectives | Separate experiment per new objective (styles may not transfer). Shadow first. Smaller confirmatory tests are acceptable if the matcher shows comparable offline quality there |
| **Roll back** | Any guardrail breach, fallback rate > 1%, latency guardrail breached, an SRM caused by the treatment, or matcher drift alarms not fixed within a week. Rollback = allocation to 100% control, with no deploy needed |

After rollout, keep the permanent 5% holdout (doc 08) so long-term effects and baseline drift stay measurable.
