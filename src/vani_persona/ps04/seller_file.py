"""Seller data file in, one persona spec per seller out.

Accepts CSV, JSON (a list of rows, or {"sellers": [...]}) or Excel with the PS07 seller-file
columns (any subset; missing fields lower confidence) plus optional use-case columns
(``use_case``, ``profile_missing_fields``, ``enquiry_product``, ``unreplied_enquiries``,
``onboarding_steps_pending``, ``subscription_plan``, ``ticket_summary``,
``listings_to_verify``, ...; see ``cook.py`` for the full list).

    python -m vani_persona.ps04.seller_file samples/cooked_sellers.csv [--out specs.jsonl]
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import pandas as pd

from vani_persona import REPO_ROOT
from vani_persona.ps04.evidence import load_evidence
from vani_persona.ps04.generator import generate
from vani_persona.ps04.use_cases import apply_use_case

SAMPLE = REPO_ROOT / "samples" / "cooked_sellers.csv"


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):          # numpy scalar
        v = v.item()
    return v


def parse(data: bytes, filename: str = "sellers.csv") -> list[dict]:
    name = filename.lower()
    if name.endswith(".json"):
        obj = json.loads(data.decode("utf-8-sig"))
        rows = obj.get("sellers", []) if isinstance(obj, dict) else obj
        df = pd.DataFrame(rows)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    for i, r in enumerate(rows, 1):
        r.setdefault("fk_glusr_usr_id", None)
        r["_key"] = str(int(r["fk_glusr_usr_id"])) if isinstance(r["fk_glusr_usr_id"], (int, float)) else f"row{i}"
    return rows


def build_spec(row: dict, evidence: dict | None = None, use_case: str | None = None) -> dict:
    """Seller row -> generator spec (voice, language, tone, plan) -> fitted to the call's use case."""
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    return apply_use_case(generate(clean, evidence or load_evidence()), clean, use_case)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, nargs="?", default=SAMPLE)
    ap.add_argument("--out", type=Path, help="write one spec per line (JSONL)")
    args = ap.parse_args()
    rows = parse(args.file.read_bytes(), args.file.name)
    ev = load_evidence()
    specs = [build_spec(r, ev) for r in rows]
    for r, s in zip(rows, specs):
        uc = s.get("use_case")
        if not uc:
            print(f"{r['_key']:>12}  do-not-call: no persona")
            continue
        print(f"{r['_key']:>12}  {uc['name']:<40} {uc['archetype']['label']:<28} start: {uc['start_persona']:<20} "
              f"{s['voice']['speaker']['value']}, pace {s['voice']['pace']['value']}, {s['language']['primary']['value']}")
    if args.out:
        args.out.write_text("\n".join(json.dumps(s, ensure_ascii=False, default=str) for s in specs), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
