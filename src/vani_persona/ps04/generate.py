"""Generate a persona spec for every seller in the seller file.

    python -m vani_persona.ps04.generate .
        -> artifacts/ps04/persona_specs.jsonl.gz   (one spec per seller; gitignored)
        -> artifacts/ps04/shared_policy.json       (facts + guardrails shared by every spec)
        -> artifacts/ps04/demo_sellers.json        (3 contrasting sellers for the voice demo)
        -> docs/findings/ps04_generator_report.md  (aggregates: types, settings, distinctness)
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import json
import random
from pathlib import Path

from vani_persona import REPO_ROOT
from vani_persona.ps04.data import load_sellers
from vani_persona.ps04.evidence import load_evidence
from vani_persona.ps04.generator import generate

OUT_DIR = REPO_ROOT / "artifacts" / "ps04"
KEY_SETTINGS = ["speaker", "pace", "language", "mix", "formality", "warmth", "empathy", "opening_len", "first_objection"]


def key_vector(spec: dict) -> dict:
    return {
        "speaker": spec["voice"]["speaker"]["value"],
        "pace": spec["voice"]["pace"]["value"],
        "language": spec["language"]["primary"]["value"],
        "mix": spec["language"]["hinglish_mix"]["value"],
        "formality": spec["language"]["formality"]["value"],
        "warmth": spec["tone"]["warmth"]["value"],
        "empathy": spec["tone"]["empathy"]["value"],
        "opening_len": len(spec["conversation_plan"]["opening"]["steps"]),
        "first_objection": spec["conversation_plan"]["objection_playbook"][0]["objection"],
    }


def _jsonable(v):
    if isinstance(v, float) and v != v:
        return None
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    evidence = load_evidence()
    sellers = load_sellers(args.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    types = collections.Counter()
    conf = collections.Counter()
    setting_counts = {k: collections.Counter() for k in KEY_SETTINGS}
    vectors, by_type = [], collections.defaultdict(list)
    n_missing = collections.Counter()
    shared = {"facts": None, "guardrails": None}
    with gzip.open(OUT_DIR / "persona_specs.jsonl.gz", "wt") as fh:
        for row in sellers.to_dict("records"):
            row = {k: _jsonable(v) for k, v in row.items()}
            spec = generate(row, evidence)
            if "voice" in spec:
                # Facts and guardrails are identical for every seller: store them once.
                shared["facts"], shared["guardrails"] = spec.pop("facts"), spec["conversation_plan"]["guardrails"]
                spec["conversation_plan"]["guardrails"] = "see shared_policy.json"
            fh.write(json.dumps(spec, ensure_ascii=False, default=str) + "\n")
            if "voice" in spec:
                spec["conversation_plan"]["guardrails"] = shared["guardrails"]
                spec["facts"] = shared["facts"]
            types[spec["seller_type"]] += 1
            if "voice" not in spec:
                continue
            conf[spec["overall_confidence"]] += 1
            for f in spec["missing_fields"]:
                n_missing[f] += 1
            v = key_vector(spec)
            vectors.append(v)
            by_type[spec["seller_type"]].append((row, spec))
            for k in KEY_SETTINGS:
                setting_counts[k][v[k]] += 1

    (OUT_DIR / "shared_policy.json").write_text(json.dumps(shared, indent=2, ensure_ascii=False), encoding="utf-8")
    rng = random.Random(7)
    sample = rng.sample(vectors, min(len(vectors), 4000))
    diffs = [sum(a[k] != b[k] for k in KEY_SETTINGS) for a, b in zip(sample[::2], sample[1::2])]
    combos = collections.Counter(tuple(v[k] for k in KEY_SETTINGS) for v in vectors)

    # Pairwise differences between the most common spec of each seller type.
    proto = {}
    for st, items in by_type.items():
        c = collections.Counter(tuple(key_vector(s)[k] for k in KEY_SETTINGS) for _, s in items)
        proto[st] = dict(zip(KEY_SETTINGS, c.most_common(1)[0][0]))
    type_pairs = [(a, b, sum(proto[a][k] != proto[b][k] for k in KEY_SETTINGS))
                  for a, b in itertools.combinations(sorted(proto), 2)]

    # Demo sellers: complete records from three contrasting types.
    demo = {}
    for st in ["new_receptive", "time_pressed", "established_formal", "guarded", "already_engaged"]:
        full = [s for r, s in by_type.get(st, []) if s["overall_confidence"] == "high"
                and len(s["conversation_plan"]["personalisation"]) >= 3]
        if full:
            demo[st] = rng.choice(full)
    south = [s for _, s in by_type.get("established_formal", []) + by_type.get("standard", [])
             if s["language"]["primary"]["value"] == "en-IN" and s["overall_confidence"] != "low"]
    if south:
        demo["english_first"] = rng.choice(south)
    (OUT_DIR / "demo_sellers.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    n = sum(types.values())
    lines = ["# PS04 persona generator report", "",
             f"Generated by `python -m vani_persona.ps04.generate .` over all **{n:,} sellers** in the seller file. "
             "Aggregates only; per-seller specs are in `artifacts/ps04/persona_specs.jsonl.gz` (gitignored).", "",
             "## Seller types (for explaining the logic; every seller still gets their own spec)", "",
             "| Type | Sellers | Share |", "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {v / n:.1%} |" for k, v in types.most_common()]
    lines += ["", "## Spec confidence (FAQ Q6: missing fields lower confidence)", "",
              "| Confidence | Sellers |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in conf.most_common()]
    lines += ["", "| Missing input | Sellers |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in n_missing.most_common()]
    lines += ["", "## Setting distributions", ""]
    for k in KEY_SETTINGS:
        lines.append(f"- **{k}**: " + ", ".join(f"{val} {c / len(vectors):.0%}" for val, c in setting_counts[k].most_common(6)))
    lines += ["", "## Distinctness", "",
              f"- Distinct combinations of the {len(KEY_SETTINGS)} key settings: **{len(combos):,}**.",
              f"- Random pairs of sellers differ on **{sum(diffs) / len(diffs):.1f}** of {len(KEY_SETTINGS)} key settings on average; "
              f"{sum(d >= 3 for d in diffs) / len(diffs):.0%} differ on 3 or more.",
              "- Typical spec per type, settings that differ between types:", "",
              "| Type A | Type B | Settings that differ |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {d} / {len(KEY_SETTINGS)} |" for a, b, d in type_pairs]
    lines += ["", "| Type | " + " | ".join(KEY_SETTINGS) + " |", "|---|" + "---|" * len(KEY_SETTINGS)]
    lines += [f"| {st} | " + " | ".join(str(p[k]) for k in KEY_SETTINGS) + " |" for st, p in sorted(proto.items())]
    out = REPO_ROOT / "docs" / "findings" / "ps04_generator_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{n:,} specs -> {OUT_DIR / 'persona_specs.jsonl.gz'}; demo sellers: {list(demo)}; report -> {out}")


if __name__ == "__main__":
    main()
