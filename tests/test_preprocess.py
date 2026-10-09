import numpy as np
import pandas as pd

from vani_persona.segmentation.preprocess import StrataResidualizer


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "campaign_id": rng.choice(["a", "b", "rare"], n, p=[0.5, 0.45, 0.05]),
            "call_objective": "obj",
            "lead_type": "PIM",
            "language_hint": rng.choice(["hi", "en"], n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    df.loc[df.campaign_id == "a", "x"] += 3  # campaign effect that residualization must remove
    df.loc[rng.random(n) < 0.1, "y"] = np.nan
    return df


def test_residualization_removes_stratum_means():
    df = frame()
    r = StrataResidualizer(["x", "y"], min_stratum_n=50).fit(df)
    z = pd.DataFrame(r.transform(df), columns=["x", "y"])
    assert abs(z[df.campaign_id == "a"]["x"].mean()) < 0.1 and abs(z[df.campaign_id == "b"]["x"].mean()) < 0.1


def test_transform_one_matches_batch_transform_including_backoff_and_missing():
    df = frame()
    r = StrataResidualizer(["x", "y"], min_stratum_n=50).fit(df)
    batch = r.transform(df)
    for i in range(0, len(df), 7):
        row = df.iloc[i]
        one = r.transform_one(
            {"x": row.x, "y": None if pd.isna(row.y) else row.y},
            {c: row[c] for c in ("campaign_id", "call_objective", "lead_type", "language_hint")},
        )
        np.testing.assert_allclose(one, batch[i], rtol=1e-9, atol=1e-12)
    # unseen stratum backs off to global means
    assert np.all(np.isfinite(r.transform_one({"x": 0.0, "y": 0.0}, {"campaign_id": "new"})))
