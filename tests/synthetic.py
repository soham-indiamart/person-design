"""SYNTHETIC test data. It exercises the pipeline code paths and is NEVER evidence about real sellers.

Three artificial behaviour styles with planted lexical signals let the tests check
that the pipeline can recover planted structure. They say nothing about real
VANI calls.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

AGENT_LINES = [
    "Namaste, main VANI IndiaMART se bol rahi hoon, catalog ke baare mein baat ho sakti hai?",
    "Aapke kuch products mein photos missing hain.",
    "Kya main WhatsApp pe link bhej doon?",
    "Link mein har product ke saamne Add photo button hai.",
    "Kya aap aaj check kar payenge?",
    "Dhanyavaad.",
]

STYLES = {
    "terse": [
        "haan", "ok", "ji", "busy hoon", "baad mein", "jaldi bolo", "theek hai", "haan bhejo", "abhi nahi", "nahi",
    ],
    "inquisitive": [
        "accha ye photos lagane se kya fayda hoga aur kitne buyers aate hain is se?",
        "mujhe samjha dijiye ye kaise karna hai app mein kahan jaana padega?",
        "kaun kaun se products mein photos nahi hain aur kya main baad mein bhi add kar sakta hoon?",
        "kya ye process mobile se ho jayega ya computer chahiye hoga uske liye?",
        "aur agar photo achhi nahi aayi toh kya hoga, kya main change kar sakta hoon baad mein?",
    ],
    "sceptical": [
        "ye fraud toh nahi hai na, number kahan se mila aapko?",
        "kitna charge lagega iska, pehle bhi paise maange the",
        "kaise pata ki aap IndiaMART se hi ho, bharosa nahi hota",
        "already kar diya tha maine, phir se kyun call kar rahe ho",
        "fees kitni hai pehle ye batao",
    ],
}
OUTCOME_P = {"terse": 0.15, "inquisitive": 0.45, "sceptical": 0.25}


def make_rows(n_calls: int = 600, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_calls):
        style = rng.choice(list(STYLES))
        n_seller = rng.randint(3, 7)
        lines = []
        for j in range(n_seller):
            lines.append(f"Agent: {AGENT_LINES[min(j, len(AGENT_LINES) - 1)]}")
            lines.append(f"Seller: {rng.choice(STYLES[style])}")
        achieved = rng.random() < OUTCOME_P[style]
        rows.append(
            {
                "call_id": f"syn-{i:05d}",
                "seller_id": f"seller-{rng.randint(0, n_calls // 2)}",
                "campaign_id": rng.choice(["camp_a", "camp_b"]),
                "call_objective": "catalog_photo_completion",
                "category": rng.choice(["industrial", "textiles", "food"]),
                "language": rng.choice(["hi", "hi", "en"]),
                "call_start_time": (t0 + timedelta(minutes=97 * i)).isoformat(),
                "agent_version": "v1",
                "script_version": "s1",
                "state": "UP",
                "disposition": "completed" if achieved else rng.choice(["not interested", "callback", "hung up"]),
                "objective_achieved": "",
                "converted": "",
                "end_reason": "normal",
                "transcript": "\n".join(lines),
                "_style": style,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in rows[0] if not c.startswith("_")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path
