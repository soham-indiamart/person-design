"""Past-call evidence behind each persona-generator rule.

    python -m vani_persona.ps04.evidence .
        -> evidence/ps04_evidence.json   (aggregates only; read by the generator)
        -> docs/findings/ps04_evidence.md

Every rule in ``generator.py`` cites one key from this file. Each entry states the
comparison, the numbers, and how strong the evidence is:

- ``moderate``: an observed difference whose 95% CI excludes zero on >= 1,000 calls.
  Still observational: VANI used one voice and style, so the data shows what each
  seller type struggles with, not that a particular voice works better.
- ``weak``: a CI that excludes zero on fewer calls, a non-randomized trial, or a
  descriptive pattern.
- ``informed_guess``: no direct evidence; the reason is stated instead.

Seller-file flags (frustration, asked-if-bot, already-in-touch, do-not-call) summarize
the same Apr–Sep calls they are compared with, so those rows include the call where the
flag was raised. History rules use ``prior_*`` columns computed from earlier calls only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vani_persona import REPO_ROOT
from vani_persona.features import lexicons as lx
from vani_persona.ps04.data import add_prior_history, load_bot_calls, load_sellers, load_turns

EVIDENCE_PATH = REPO_ROOT / "evidence" / "ps04_evidence.json"

_H = lx.HINDI_FUNCTION_WORDS - lx.AMBIGUOUS_FUNCTION_WORDS
_E = lx.ENGLISH_FUNCTION_WORDS - lx.AMBIGUOUS_FUNCTION_WORDS


def _cmp(y: pd.Series, mask: pd.Series) -> dict:
    """Meeting rate inside vs outside ``mask`` with a normal-approx 95% CI on the difference."""
    a, o = y[mask], y[~mask]
    p1, p0, n1, n0 = float(a.mean()), float(o.mean()), int(len(a)), int(len(o))
    se = math.sqrt(p1 * (1 - p1) / max(n1, 1) + p0 * (1 - p0) / max(n0, 1))
    d = p1 - p0
    ci = (d - 1.96 * se, d + 1.96 * se)
    sig = ci[0] > 0 or ci[1] < 0
    strength = "moderate" if sig and n1 >= 1000 else "weak" if sig else "inconclusive"
    return {"rate_group": round(p1, 4), "rate_rest": round(p0, 4), "n_group": n1, "n_rest": n0,
            "diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "strength": strength}


def _entry(claim: str, stats: dict, source: str, strength: str | None = None, caveat: str = "") -> dict:
    return {"claim": claim, "stats": stats, "source": source,
            "strength": strength or stats.get("strength", "weak"), "caveat": caveat}


def build(root: Path) -> dict:
    s = load_sellers(root)
    b = add_prior_history(load_bot_calls(root))
    m = b.merge(s, on="fk_glusr_usr_id", how="left")
    y = m["meeting_fixed"].astype(float)
    ev: dict[str, dict] = {"_meta": {
        "answered_calls": int(len(b)), "sellers": int(len(s)), "baseline_meeting_rate": round(float(y.mean()), 4),
        "outcome": "meeting_fixed on answered VANI bot calls, Apr-Sep 2026",
    }}

    gst = pd.to_numeric(m["gst_registration_year"], errors="coerce")
    ev["new_seller"] = _entry(
        "Sellers GST-registered in 2024 or later fix meetings more often: open warmly and explain the benefit.",
        _cmp(y, gst >= 2024), "bot calls x seller file (gst_registration_year)")
    turnover_hi = m["annual_turnover"].isin(["1.5 - 5 Cr", "5 - 25 Cr", "25 - 100 Cr", "100 - 500 Cr", "> 500 Cr"])
    ev["high_turnover"] = _entry(
        "Sellers with turnover >= 1.5 Cr fix fewer meetings and hang up sooner: be concise, more formal, value-first.",
        _cmp(y, turnover_hi), "bot calls x seller file (annual_turnover)")
    ev["limited_company"] = _entry(
        "Limited companies fix fewer meetings than proprietorships: more formal register.",
        _cmp(y, m["business_type"] == "Limited Company"), "bot calls x seller file (business_type)")
    for col, key, claim in [
        ("showed_frustration", "past_frustration",
         "Sellers who showed frustration before fix fewer meetings: calm, slower, apologetic, offer a callback early."),
        ("asked_if_talking_to_bot", "asked_if_bot",
         "Sellers who asked if they were talking to a bot: say up front that this is IndiaMART's AI assistant."),
        ("already_in_touch_with_executive", "already_in_touch",
         "Sellers already in touch with an executive fix fewer meetings and talk longer: acknowledge it first."),
        ("do_not_call_requested", "do_not_call",
         "Sellers who asked not to be called almost never fix meetings: suppress the call (policy, not persona)."),
    ]:
        flag = pd.to_numeric(m[col], errors="coerce") == 1
        ev[key] = _entry(claim, _cmp(y, flag), f"bot calls x seller file ({col})",
                         caveat="Flag summarizes the same Apr-Sep calls; includes the call where it was raised.")

    known = m["prior_n"] > 0
    short = known & (m["prior_dur_mean"] < 25)
    st = _cmp(y[known], short[known])
    st["median_duration_group_s"] = float(m.loc[short, "lead_call_duration"].median())
    st["median_duration_rest_s"] = float(m.loc[known & ~short, "lead_call_duration"].median())
    ev["short_call_history"] = _entry(
        "Sellers whose earlier calls averaged < 25 s hang up early again: shorten turns and get to the point fast.",
        st, "bot calls, prior calls only (prior_dur_mean)")
    ev["prior_call_later"] = _entry(
        "Sellers who said 'call later' before: reference the callback and ask for a time early.",
        _cmp(y[known], (m["prior_call_later"] > 0)[known]), "bot calls, prior calls only")
    ev["prior_not_interested"] = _entry(
        "Sellers who said 'not interested' before: short, low-pressure, give one reason then offer to end.",
        _cmp(y[known], (m["prior_not_interested"] > 0)[known]), "bot calls, prior calls only")
    ev["first_call"] = _entry(
        "First answered call to a seller (no prior history): default settings; live adaptation corrects them.",
        _cmp(y, m["prior_n"] == 0), "bot calls, prior calls only")

    # Personalised-script trial: compare with other bot versions on the same days.
    m["day"] = m["ts"].dt.date
    pers = m["lead_bot_version"].astype(str).str.startswith("personalis")
    days = set(m.loc[pers, "day"])
    same_days = m["day"].isin(days)
    ev["personalised_trial"] = _entry(
        "The personalised-script trial fixed more meetings than other versions on the same days: personalise the opening.",
        _cmp(y[same_days], pers[same_days]), "bot calls, lead_bot_version on trial days",
        strength="weak", caveat="Not randomized; ran on few days. Treat as a hint, not proof.")

    # Language: how sellers actually spoke on transcribed calls, by state.
    t = load_turns(root)
    seller = t[t["speaker"] == "seller"].copy()
    toks = seller["text"].fillna("").astype(str).str.lower().str.findall(r"[a-z']+")
    seller["h"] = toks.apply(lambda w: sum(x in _H for x in w))
    seller["e"] = toks.apply(lambda w: sum(x in _E for x in w))
    seller["deva"] = seller["text"].fillna("").astype(str).str.contains(r"[ऀ-ॿ]")
    per = seller.groupby("attempt_id").agg(h=("h", "sum"), e=("e", "sum"), deva=("deva", "mean"))
    per = per.join(t.groupby("attempt_id")["fk_glusr_usr_id"].first()).merge(
        s[["fk_glusr_usr_id", "seller_state", "business_type", "annual_turnover"]], on="fk_glusr_usr_id", how="left")
    per["english_leaning"] = per["e"] / (per["h"] + per["e"]).replace(0, np.nan) > 0.6
    by_state = per.groupby("seller_state")["english_leaning"].agg(["size", "mean"])
    by_state = by_state[by_state["size"] >= 100].sort_values("mean", ascending=False)
    ev["language_by_state"] = _entry(
        "Share of transcribed calls where the seller spoke mostly English, by state (>= 100 calls).",
        {"overall_english_leaning": round(float(per["english_leaning"].mean()), 4),
         "devanagari_turn_share": round(float(seller["deva"].mean()), 4),
         "by_state": {k: {"n": int(r["size"]), "english_leaning": round(float(r["mean"]), 4)}
                      for k, r in by_state.iterrows()}},
        "ps07_call_turns (function-word language share)", strength="moderate",
        caveat="Transcript subset over-represents meetings; romanized text, so regional languages are undercounted.")
    biz = per.groupby("business_type")["english_leaning"].agg(["size", "mean"])
    ev["language_by_business_type"] = _entry(
        "Limited companies lean English slightly more often than proprietorships.",
        {k: {"n": int(r["size"]), "english_leaning": round(float(r["mean"]), 4)} for k, r in biz.iterrows()},
        "ps07_call_turns", strength="weak")

    # Pricing consistency in today's VANI: different calls quote different prices.
    bot = t[t["speaker"] == "bot"]["text"].fillna("").astype(str).str.lower()
    price_lines = bot[bot.str.contains(r"(?:rupees|rs\.?|₹|hazaar|thousand|\d{1,3},\d{3}).{0,40}(?:gst|mahina|month|annual|year|plan)", regex=True)]
    amounts = sorted({a for line in price_lines for a in re.findall(r"\d{1,3}(?:,\d{3})+|\d+\s*-\s*\d+\s*hazaar", line)})
    ev["price_inconsistency"] = _entry(
        "Today's VANI states different prices on different calls: personas never state prices; the executive does.",
        {"bot_turns_with_a_price": int(len(price_lines)), "distinct_amounts_quoted": amounts[:12]},
        "ps07_call_turns, bot turns", strength="moderate")

    # Facts VANI already says (the persona may rephrase delivery but not these).
    ev["identity_drop"] = _entry(
        "17.3% of evaluated calls drop right after the IndiaMART / identity moment and 25.2% never respond: "
        "keep the first turn short, check the line, and state purpose and benefit together.",
        {"drop_after_identity_share": 0.173, "never_respond_share": 0.252},
        "disposition evaluator, docs/findings/dataset_overview.md", strength="moderate")
    ev["bot_latency"] = _entry(
        "Calls where the bot's median reply lag is <= 1.5 s fix ~25% vs 4.6% above 3 s: adaptation must add no delay.",
        {"mf_lag_le_1_5s": 0.25, "mf_lag_gt_2s": 0.138, "mf_lag_gt_3s": 0.046},
        "latency_repetition evaluator, docs/findings/dataset_overview.md", strength="moderate",
        caveat="Correlational; long lag also follows long seller turns.")
    for k in ("voice_gender", "voice_speaker", "pitch"):
        ev[k] = _entry(
            {"voice_gender": "VANI has always used one female voice ('Payal ... bol rahi hoon'); no data compares genders.",
             "voice_speaker": "Speaker per language follows Sarvam's quality ranking; tone-to-speaker mapping is a design choice.",
             "pitch": "No pitch variation exists in past calls."}[k],
            {}, "none in the export", strength="informed_guess")
    return ev


def to_markdown(ev: dict) -> str:
    meta = ev["_meta"]
    lines = [
        "# PS04 evidence: past calls behind each generator rule", "",
        f"Generated by `python -m vani_persona.ps04.evidence .`. Outcome: {meta['outcome']}. "
        f"{meta['answered_calls']:,} answered calls, {meta['sellers']:,} sellers, baseline meeting rate "
        f"{meta['baseline_meeting_rate']:.1%}. Aggregates only.", "",
        "Strength: **moderate** = 95% CI excludes zero on >= 1,000 calls (still observational); "
        "**weak** = smaller, non-randomized or descriptive; **informed_guess** = no direct evidence.", "",
        "| Key | Claim | Group rate | Rest | n (group) | 95% CI of diff | Strength |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, e in ev.items():
        if k.startswith("_"):
            continue
        st = e["stats"]
        if "rate_group" in st:
            lines.append(f"| `{k}` | {e['claim']} | {st['rate_group']:.1%} | {st['rate_rest']:.1%} | "
                         f"{st['n_group']:,} | {st['ci95'][0]:+.1%} to {st['ci95'][1]:+.1%} | {e['strength']} |")
        else:
            lines.append(f"| `{k}` | {e['claim']} | | | | | {e['strength']} |")
    lines += ["", "## Language by state (transcribed calls)", "",
              "| State | Calls | Seller mostly English |", "|---|---|---|"]
    for state, r in ev["language_by_state"]["stats"]["by_state"].items():
        lines.append(f"| {state} | {r['n']:,} | {r['english_leaning']:.1%} |")
    lines += ["", "## Caveats", ""]
    lines += [f"- `{k}`: {e['caveat']}" for k, e in ev.items() if not k.startswith("_") and e.get("caveat")]
    return "\n".join(lines) + "\n"


def load_evidence(path: Path = EVIDENCE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    ev = build(args.root)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    out = REPO_ROOT / "docs" / "findings" / "ps04_evidence.md"
    out.write_text(to_markdown(ev), encoding="utf-8")
    print(f"wrote {EVIDENCE_PATH} and {out}")


if __name__ == "__main__":
    main()
