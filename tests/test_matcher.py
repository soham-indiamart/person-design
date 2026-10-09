from vani_persona.matching.matcher import MatchRequest, PersonaMatcher, PrototypeScorer
from vani_persona.personas import NEUTRAL_ID


def feats(**kw):
    base = {"n_seller_turns": 3.0}
    base.update(kw)
    return base


RESIST = dict(short_reply_ratio=0.33, question_rate=0.3, affirmation_rate=0.2, log_mean_tokens=2.4,
              resistance_rate=0.4, rush_rate=0.0)
RUSH = dict(short_reply_ratio=0.4, question_rate=0.05, affirmation_rate=0.3, log_mean_tokens=2.0,
            resistance_rate=0.0, rush_rate=0.5)


def make(registry, **kw):
    return PersonaMatcher(PrototypeScorer(registry, temperature=0.5), **{"min_seller_turns": 2, "commit_threshold": 0.6, **kw})


def test_scores_are_a_distribution_including_neutral(registry):
    r = make(registry).match(MatchRequest("c", 3, feats(**RESIST), {}))
    assert abs(sum(r.scores.values()) - 1) < 1e-3 and NEUTRAL_ID in r.scores
    assert r.calibrated is False and r.scorer == "prototype"


def test_resistant_and_rushed_profiles_pick_different_personas(registry):
    m = make(registry)
    assert m.match(MatchRequest("c", 3, feats(**RESIST), {})).best_persona_id == "objection_raising"
    assert m.match(MatchRequest("c", 3, feats(**RUSH), {})).best_persona_id == "time_pressed"


def test_exceeding_a_distinctive_feature_is_not_penalized(registry):
    m = make(registry)
    moderate = m.match(MatchRequest("c", 3, feats(**{**RESIST, "resistance_rate": 0.2}), {}))
    extreme = m.match(MatchRequest("c", 3, feats(**{**RESIST, "resistance_rate": 0.8}), {}))
    assert extreme.scores["objection_raising"] >= moderate.scores["objection_raising"] - 1e-9


def test_abstains_with_too_few_turns(registry):
    r = make(registry).match(MatchRequest("c", 1, {**RESIST, "n_seller_turns": 1.0}, {}))
    assert r.abstained and r.best_persona_id == NEUTRAL_ID and "insufficient_evidence" in r.reason


def test_abstains_when_nothing_fits(registry):
    r = make(registry).match(MatchRequest("c", 3, feats(), {}))  # no behavioural features at all
    assert r.abstained and r.best_persona_id == NEUTRAL_ID


def test_scorer_failure_is_contained(registry):
    class Boom:
        name, calibrated, version = "prototype", False, "x"

        def score(self, req):
            raise RuntimeError("model down")

    r = PersonaMatcher(Boom()).match(MatchRequest("c", 3, feats(**RESIST), {}, current_persona_id="objection_raising"))
    assert r.abstained and r.scores == {} and r.best_persona_id == "objection_raising" and "scorer_error" in r.reason
