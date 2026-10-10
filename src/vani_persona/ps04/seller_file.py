"""Seller data file in, one persona spec per seller out.

Accepts CSV / TSV / any delimited text, JSON (a list, {"sellers": [...]}, nested objects, one
seller) or JSON Lines, Excel and XML; the format is detected from the content. Columns are the
PS07 seller-file ones (any subset; common aliases like ``id`` / ``state`` / ``company`` are mapped;
missing fields lower confidence) plus optional use-case columns
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


# Other names a CRM / sheet export may use for the PS07 columns (after lower-casing, non-alphanumerics -> "_").
ALIASES = {
    "fk_glusr_usr_id": ["glusr_usr_id", "glusr_id", "glid", "seller_id", "sellerid", "user_id", "id"],
    "seller_state": ["state", "seller_state_name", "region_state", "address_state"],
    "seller_city": ["city", "address_city"],
    "seller_name": ["name", "contact_name", "contact_person", "owner_name", "owner", "seller"],
    "company_name": ["company", "business_name", "firm_name", "firm", "seller_company"],
    "use_case": ["usecase", "use_case_id", "call_type", "call_purpose", "purpose", "journey", "campaign"],
    "top_category_1": ["category", "main_category", "product_category", "top_category"],
    "business_type": ["business_constitution", "constitution"],
    "nature_of_business": ["seller_type_name", "business_nature"],
    "subscription_plan": ["plan", "current_plan"],
}
LIST_KEYS = ("sellers", "data", "rows", "records", "items", "results")
FORMATS = {"excel": "Excel", "json": "JSON", "jsonl": "JSON Lines", "xml": "XML", "csv": "CSV", "tsv": "TSV",
           "delimited": "delimited text"}


def _text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def detect_format(data: bytes, filename: str = "") -> str:
    """Format from the content first (a mislabelled file still works), the extension as a tie-break."""
    if data[:4] == b"PK\x03\x04" or data[:4] == b"\xd0\xcf\x11\xe0":
        return "excel"
    head = _text(data[:4096]).lstrip()
    if head[:1] in "[{":
        try:
            json.loads(_text(data))
            return "json"
        except json.JSONDecodeError:
            return "jsonl"
    if head[:1] == "<":
        return "xml"
    first = head.splitlines()[0] if head else ""
    if filename.lower().endswith(".tsv") or first.count("\t") > max(first.count(","), 0):
        return "tsv"
    return "csv" if first.count(",") >= max(first.count(";"), first.count("|")) else "delimited"


def _records(obj) -> list[dict]:
    """Seller rows from any JSON shape: a list, {"sellers": [...]}, {"data": {...: [...]}}, or one seller."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if isinstance(obj.get(k), (list, dict)):
                return _records(obj[k])
        lists = [v for v in obj.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
        if len(lists) == 1:
            return lists[0]
        return [obj]
    return []


def _snake(c) -> str:
    return "_".join("".join(ch if ch.isalnum() else " " for ch in str(c).lower()).split())


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Snake-case headers and fold alias columns into the PS07 ones (rows may use different aliases)."""
    df = df.rename(columns=_snake)
    df = df.loc[:, ~df.columns.duplicated()]
    for canon, alts in ALIASES.items():
        cols = [c for c in [canon, *alts] if c in df.columns]
        if not cols or cols == [canon]:
            continue
        merged = df[cols[0]]
        for c in cols[1:]:
            merged = merged.combine_first(df[c])
        df = df.drop(columns=[c for c in cols if c != canon]).assign(**{canon: merged})
    return df


def _frame(data: bytes, fmt: str) -> pd.DataFrame:
    if fmt == "excel":
        return pd.read_excel(io.BytesIO(data))
    text = _text(data)
    if fmt == "json":
        return pd.json_normalize(_records(json.loads(text)), sep="_")
    if fmt == "jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.json_normalize([r for r in rows if isinstance(r, dict)], sep="_")
    if fmt == "xml":
        return pd.read_xml(io.StringIO(text), parser="etree")
    sep = {"csv": ",", "tsv": "\t"}.get(fmt)
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python" if sep is None else "c")


def _key(v, i: int) -> str:
    if isinstance(v, bool) or v is None:
        return f"row{i}"
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        s = "".join(ch for ch in str(v).strip() if ch.isalnum() or ch in "-_")
        return s or f"row{i}"


def parse(data: bytes, filename: str = "sellers.csv") -> list[dict]:
    """Seller rows from CSV / TSV / any delimited text, JSON (any shape) / JSON Lines, Excel or XML."""
    df = _frame(data, detect_format(data, filename))
    df = _canonical(df).dropna(how="all")
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    seen: set[str] = set()
    for i, r in enumerate(rows, 1):
        r.setdefault("fk_glusr_usr_id", None)
        key = _key(r["fk_glusr_usr_id"], i)
        r["_key"] = key if key not in seen else f"{key}-{i}"
        seen.add(r["_key"])
        if isinstance(r["fk_glusr_usr_id"], str) and r["fk_glusr_usr_id"].strip().isdigit():
            r["fk_glusr_usr_id"] = int(r["fk_glusr_usr_id"].strip())
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
