"""Replay historical calls turn by turn through the live CallSession (shadow mode).

    python -m vani_persona.matching.replay . --source ps07

The trained matcher and switching state machine run exactly as they would live.
The report gives commit timing, switch and revert counts, oscillation, and
agreement between the persona active at the end of the call and the call's
full-call segment. It is the Phase 3 replay check (docs/11-roadmap.md). Agreement
here is with cluster labels, not with human review.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from vani_persona import REPO_ROOT
from vani_persona.canonical import Speaker
from vani_persona.config import load_thresholds
from vani_persona.events import EventLog
from vani_persona.ingest.sources import SOURCES, load_calls_from
from vani_persona.matching.matcher import default_matcher
from vani_persona.matching.train import temporal_split
from vani_persona.orchestration.dialogue_state import ObjectivePolicy
from vani_persona.orchestration.session import CallSession
from vani_persona.personas import NEUTRAL_ID, PersonaRegistry
from vani_persona.segmentation.preprocess import build_frame
from vani_persona.switching.state_machine import Action, SwitchConfig

log = logging.getLogger(__name__)


def replay(raw: Path, source: str, segments_csv: Path, artifact: Path, objective: str, test_only: bool = True) -> dict:
    th = load_thresholds()
    registry = PersonaRegistry()
    policy = ObjectivePolicy.load(objective)
    matcher = default_matcher(registry, th, artifact)
    seg_to_persona = {registry.get(p)["matching"]["segment_id"]: p for p in registry.ids(include_neutral=True)}
    labels = pd.read_csv(segments_csv, dtype={"call_id": str}).set_index("call_id")["segment"]
    calls = [c for c in load_calls_from(raw, source) if c.call_id in labels.index]
    if test_only:
        frame = build_frame(calls, max_seller_turns=1)
        _, test = temporal_split(frame)
        keep = set(frame.loc[test, "call_id"])
        calls = [c for c in calls if c.call_id in keep]

    rows = []
    for call in calls:
        s = CallSession(
            call.call_id, policy, registry, matcher, SwitchConfig.from_thresholds(th), mode="shadow",
            context={"campaign_id": call.context.campaign_id, "call_objective": call.context.call_objective,
                     "lead_type": call.context.lead_type, "language_hint": call.context.language_hint},
            events=EventLog(call.call_id, call.call_id, None),
            recent_window=th["matching"]["recent_window_seller_turns"],
        )
        for t in call.turns:
            if t.speaker == Speaker.AGENT:
                s.on_agent_turn(t.text)
            elif t.speaker == Speaker.SELLER:
                r = s.on_seller_turn(t.text)
                if r.end_call:
                    break
        hist = s.switcher.state.history
        changes = [d for d in hist if d.action != Action.RETAIN]
        first_commit = next((d.seller_turn for d in hist if d.action == Action.COMMIT), None)
        visited = [d.to_persona for d in changes]
        rows.append(
            {
                "call_id": call.call_id,
                "seller_turns": s.seller_turns,
                "first_commit_turn": first_commit,
                "n_changes": len(changes),
                "n_switches": sum(d.action == Action.SWITCH for d in changes),
                "n_reverts": sum(d.action == Action.REVERT_TO_NEUTRAL for d in changes),
                "returned_to_previous": any(visited[i] in visited[:i - 1] for i in range(2, len(visited))),
                "final_persona": s.switcher.state.active,
                "true_persona": seg_to_persona.get(int(labels[call.call_id]), NEUTRAL_ID),
                "max_match_ms": max((e["payload"]["latency_ms"] for e in s.events.events if e["event_type"] == "match_evaluated"), default=0.0),
            }
        )
    df = pd.DataFrame(rows)
    return {"df": df, "scorer": matcher.scorer.name, "model_version": matcher.scorer.version}


def write_report(res: dict, path: Path) -> None:
    df = res["df"]
    n = len(df)
    agree = (df.final_persona == df.true_persona)
    lines = [
        "# Replay report (generated)",
        "",
        f"{n} held-out calls replayed turn by turn through `CallSession(mode=\"shadow\")` with scorer "
        f"`{res['scorer']}` (`{res['model_version']}`) and the thresholds in `config/thresholds.yaml`. "
        "Agreement is measured against full-call cluster labels, not human review.",
        "",
        "## Switching behaviour",
        "",
        f"- Calls that never left neutral: {(df.n_changes == 0).mean():.1%}",
        f"- First commit at seller turn: {df.first_commit_turn.dropna().astype(int).value_counts(normalize=True).sort_index().round(3).to_dict()}",
        f"- Persona changes per call: {df.n_changes.value_counts(normalize=True).sort_index().round(3).to_dict()}",
        f"- Calls with > 1 change: {(df.n_changes > 1).mean():.1%} (Phase 3 target ≤ 5%, counting the first commit as a change)",
        f"- Calls with a switch between two personas: {(df.n_switches > 0).mean():.1%}; with a revert to neutral: {(df.n_reverts > 0).mean():.1%}",
        f"- Calls that returned to an earlier persona (possible oscillation): {df.returned_to_previous.mean():.1%}",
        f"- Max matcher latency within a call, p50 / p99: {df.max_match_ms.quantile(0.5):.2f} / {df.max_match_ms.quantile(0.99):.2f} ms (this machine)",
        "",
        "## Final persona vs full-call segment",
        "",
        f"- Agreement: {agree.mean():.1%}",
        "",
        "| true (segment persona) | calls | final = true | final distribution |",
        "|---|---|---|---|",
    ]
    for p, g in df.groupby("true_persona"):
        dist = Counter(g.final_persona)
        lines.append(
            f"| {p} | {len(g)} | {(g.final_persona == p).mean():.1%} | "
            + ", ".join(f"{k} {v / len(g):.0%}" for k, v in dist.most_common())
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, nargs="?", default=REPO_ROOT / "data" / "raw")
    p.add_argument("--source", choices=SOURCES, default="generic")
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "derived" / "segments.csv")
    p.add_argument("--artifact", type=Path, default=REPO_ROOT / "artifacts" / "matcher.joblib")
    p.add_argument("--objective", default="fix_meeting")
    p.add_argument("--all", action="store_true", help="replay all labelled calls, not just the temporal test split")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "findings" / "replay_report.md")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    res = replay(a.raw, a.source, a.segments, a.artifact, a.objective, test_only=not a.all)
    write_report(res, a.out)
    print(f"replayed {len(res['df'])} calls -> {a.out}")


if __name__ == "__main__":
    main()
