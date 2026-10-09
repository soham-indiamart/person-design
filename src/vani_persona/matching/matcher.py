"""Persona matching: features -> persona probabilities + evidence + abstention.

Two interchangeable scorers:
  LogisticScorer  - calibrated multinomial logistic regression trained on segment
                    labels (``matching.train``). Recommended once data exists.
  PrototypeScorer - distance to each persona's ``matching.prototype``. Needs no
                    training, but the output is NOT calibrated. Used to bootstrap
                    before a model exists, and as a fallback when the model artifact
                    is missing.
An LLM is never the classifier.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from vani_persona.features.registry import FEATURES
from vani_persona.personas import NEUTRAL_ID, PersonaRegistry


@dataclass
class MatchRequest:
    call_id: str
    turn_index: int
    features: dict[str, float | None]
    context: dict[str, str | None]
    current_persona_id: str = NEUTRAL_ID


@dataclass
class Evidence:
    feature: str
    value: float | None
    detail: str


@dataclass
class MatchResult:
    call_id: str
    turn_index: int
    scores: dict[str, float]
    best_persona_id: str
    confidence: float
    calibrated: bool
    abstained: bool
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    uncertain_signals: list[str] = field(default_factory=list)
    scorer: str = ""
    model_version: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Scorer(Protocol):
    name: str
    calibrated: bool
    version: str

    def score(self, req: MatchRequest) -> tuple[dict[str, float], list[Evidence]]: ...


class PrototypeScorer:
    """softmax over -d²/2T, where d² is the mean squared standardized distance to each prototype.

    Neutral takes part as a reference class with d² = ``neutral_d2``: a persona only
    beats neutral if it fits better than "one scale unit away on average".

    Only the persona's declared distinctive features are used, one-sided. A seller who
    objects MORE than the objection_raising prototype is not penalized for it. On PS07
    the segments are defined by one distinctive feature each; including the
    non-distinctive features made clearly resistant sellers score as neutral. If a
    persona declares no distinctive features, all of its prototype features are used,
    two-sided.
    """

    name = "prototype"
    calibrated = False

    def __init__(self, registry: PersonaRegistry, temperature: float = 1.0, neutral_d2: float = 1.0):
        self.temperature = temperature
        self.neutral_d2 = neutral_d2
        self.protos: dict[str, dict] = {}
        self.directions: dict[str, dict[str, str]] = {}
        for pid in registry.ids():
            p = registry.get(pid)
            if p["matching"].get("prototype"):
                self.protos[pid] = p["matching"]["prototype"]
                self.directions[pid] = {d["feature"]: d["direction"] for d in p["definition"]["distinctive_features"]}
        self.version = "prototype:" + ",".join(f"{p}@{registry.get(p)['version']}" for p in sorted(self.protos))

    def _d2(self, pid: str, feats: dict) -> tuple[float | None, list[tuple[str, float]]]:
        zs = []
        directions = self.directions[pid]
        for f, p in self.protos[pid].items():
            if feats.get(f) is None or (directions and f not in directions):
                continue
            z = (feats[f] - p["center"]) / p["scale"]
            direction = self.directions[pid].get(f)
            if (direction == "high" and z > 0) or (direction == "low" and z < 0):
                z = 0.0
            zs.append((f, z))
        if not zs:
            return None, []
        return sum(z * z for _, z in zs) / len(zs), zs

    def score(self, req: MatchRequest) -> tuple[dict[str, float], list[Evidence]]:
        logits = {NEUTRAL_ID: -self.neutral_d2 / (2 * self.temperature)}
        zmap = {}
        for pid in self.protos:
            d2, zs = self._d2(pid, req.features)
            if d2 is not None:
                logits[pid] = -d2 / (2 * self.temperature)
                zmap[pid] = zs
        m = max(logits.values())
        exp = {k: math.exp(v - m) for k, v in logits.items()}
        tot = sum(exp.values())
        scores = {k: v / tot for k, v in exp.items()}
        best = max((k for k in scores if k != NEUTRAL_ID), key=scores.get, default=None)
        evidence = [
            Evidence(f, req.features.get(f), f"{z:+.2f} scale units off the {best} prototype")
            for f, z in sorted(zmap.get(best, []), key=lambda t: abs(t[1]))
        ]
        return scores, evidence


class LogisticScorer:
    """Loads ``artifacts/matcher.joblib`` produced by ``matching.train``."""

    name = "logistic"
    calibrated = True

    def __init__(self, artifact: Path, registry: PersonaRegistry):
        art = joblib.load(artifact)
        self.resid = art["residualizer"]
        self.model = art["calibrated_model"]
        self.explainer = art["explainer"]
        self.features = art["features"]
        self.version = art["version"]
        seg_to_persona = {
            registry.get(pid)["matching"]["segment_id"]: pid
            for pid in registry.ids()
            if registry.get(pid)["matching"]["segment_id"] is not None
        }
        self.class_to_persona = [seg_to_persona.get(int(c)) for c in self.model.classes_]

    def score(self, req: MatchRequest) -> tuple[dict[str, float], list[Evidence]]:
        x = self.resid.transform_one(req.features, req.context)
        probs = self.model.predict_proba(x.reshape(1, -1))[0]
        scores: dict[str, float] = {}
        for pid, p in zip(self.class_to_persona, probs):
            # Segments without an approved persona map to neutral.
            key = pid or NEUTRAL_ID
            scores[key] = scores.get(key, 0.0) + float(p)
        best_idx = int(np.argmax(probs))
        contrib = self.explainer.coef_[best_idx] * x
        order = np.argsort(-np.abs(contrib))[:5]
        evidence = [
            Evidence(self.features[i], req.features.get(self.features[i]), f"contribution {contrib[i]:+.2f} (log-odds)")
            for i in order
        ]
        return scores, evidence


class PersonaMatcher:
    def __init__(self, scorer: Scorer, min_seller_turns: int = 2, commit_threshold: float = 0.6):
        self.scorer = scorer
        self.min_seller_turns = min_seller_turns
        self.commit_threshold = commit_threshold

    def match(self, req: MatchRequest) -> MatchResult:
        t0 = time.perf_counter()
        uncertain = [
            f for f, spec in FEATURES.items()
            if spec.role == "segment" and (req.features.get(f) is None or spec.reliability == "low")
        ]
        try:
            scores, evidence = self.scorer.score(req)
        except Exception as e:  # never let matching break the call
            return MatchResult(
                req.call_id, req.turn_index, {}, req.current_persona_id, 0.0, False, True,
                f"scorer_error: {type(e).__name__}", [], uncertain, self.scorer.name,
                getattr(self.scorer, "version", ""), (time.perf_counter() - t0) * 1000,
            )
        candidates = {k: v for k, v in scores.items() if k != NEUTRAL_ID}
        best = max(candidates, key=candidates.get) if candidates else NEUTRAL_ID
        conf = candidates.get(best, 0.0)
        n_turns = int(req.features.get("n_seller_turns") or 0)
        if n_turns < self.min_seller_turns:
            abstained, reason = True, f"insufficient_evidence: {n_turns} < {self.min_seller_turns} seller turns"
        elif conf < self.commit_threshold:
            abstained, reason = True, f"low_confidence: {best} {conf:.2f} < {self.commit_threshold}"
        else:
            abstained, reason = False, f"{best} {conf:.2f} >= {self.commit_threshold}"
        return MatchResult(
            call_id=req.call_id,
            turn_index=req.turn_index,
            scores={k: round(v, 4) for k, v in scores.items()},
            best_persona_id=NEUTRAL_ID if abstained else best,
            confidence=round(conf, 4),
            calibrated=self.scorer.calibrated,
            abstained=abstained,
            reason=reason,
            evidence=evidence,
            uncertain_signals=uncertain,
            scorer=self.scorer.name,
            model_version=self.scorer.version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )


def default_matcher(registry: PersonaRegistry, thresholds: dict, artifact: Path | None = None) -> PersonaMatcher:
    """Use the trained model if present, otherwise fall back to prototypes."""
    m = thresholds["matching"]
    if artifact is not None and artifact.exists():
        scorer: Scorer = LogisticScorer(artifact, registry)
    else:
        scorer = PrototypeScorer(registry, temperature=m["prototype_temperature"])
    return PersonaMatcher(scorer, m["min_seller_turns_before_commit"], m["commit_threshold"])
