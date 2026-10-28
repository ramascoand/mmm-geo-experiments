"""Simulated marketing data with known ground truth.

One "world" is used across the notebooks. Sales are generated at the geo-week
level, so the same data can be aggregated to a national weekly series (for the
MMM) or kept as a geo panel (for the holdout experiment).

Media effects follow the same functional form used by PyMC-Marketing:
geometric adstock (normalized weights) followed by logistic saturation applied
to spend scaled by the channel's national maximum. Spend is partly set in
response to an unobserved demand shock, so a regression of sales on spend
overstates media effects. That is the bias the geo experiment is meant to fix.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["tv", "search", "social"]


def geometric_adstock(x, alpha, l_max=8, normalize=True):
    """Carryover: today's effective spend is a decaying sum of past spend."""
    w = alpha ** np.arange(l_max)
    if normalize:
        w = w / w.sum()
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for lag, wl in enumerate(w):
        out[lag:] += wl * x[: len(x) - lag]
    return out


def logistic_saturation(x, lam):
    """Diminishing returns, bounded between 0 and 1."""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


@dataclass
class WorldParams:
    n_weeks: int = 156
    n_geos: int = 40
    start: str = "2023-01-02"
    beta: dict = field(default_factory=lambda: {"tv": 260.0, "search": 140.0, "social": 90.0})
    lam: dict = field(default_factory=lambda: {"tv": 2.0, "search": 3.0, "social": 1.5})
    alpha: dict = field(default_factory=lambda: {"tv": 0.6, "search": 0.1, "social": 0.3})
    intercept: float = 1000.0
    trend: float = 0.8
    season_amp: float = 120.0
    promo_effect: float = 90.0
    shock_rho: float = 0.8
    shock_sd: float = 35.0
    noise_sd: float = 30.0
    geo_noise_sd: float = 12.0
    seed: int = 7


def _national_spend(p, rng, season, shock):
    t = np.arange(p.n_weeks)
    flight = ((t % 13) < 7).astype(float)
    tv = flight * (55 + 25 * season + 0.35 * shock) + rng.normal(0, 4, p.n_weeks)
    search = 30 + 8 * season + 0.25 * shock + rng.normal(0, 3, p.n_weeks)
    social = 22 + rng.gamma(2.0, 4.0, p.n_weeks)
    spend = pd.DataFrame({"tv": tv, "search": search, "social": social})
    return spend.clip(lower=0)


def simulate_world(p: WorldParams | None = None, holdout: dict | None = None):
    """Simulate the geo-week panel.

    holdout: optional dict with keys channel, geos (list of geo ids), start_week,
    end_week (inclusive). Spend on that channel is set to zero in those geos and
    weeks.

    Returns a dict with:
      panel    geo-week data (sales, spend by channel, controls, true contributions)
      national weekly aggregate in the format PyMC-Marketing expects
      truth    true ROI by channel and the true parameters
    """
    p = p or WorldParams()
    rng = np.random.default_rng(p.seed)
    t = np.arange(p.n_weeks)
    dates = pd.date_range(p.start, periods=p.n_weeks, freq="W-MON")

    season = np.sin(2 * np.pi * t / 52) + 0.5 * np.cos(2 * np.pi * t / 26)
    shock = np.zeros(p.n_weeks)
    for i in range(1, p.n_weeks):
        shock[i] = p.shock_rho * shock[i - 1] + rng.normal(0, p.shock_sd)
    promo = (rng.random(p.n_weeks) < 0.15).astype(float)

    spend = _national_spend(p, rng, season, shock)
    scale = spend.max()

    share = rng.dirichlet(np.full(p.n_geos, 8.0))

    rows = []
    for g in range(p.n_geos):
        geo_spend = spend.mul(share[g])
        if holdout is not None and g in holdout["geos"]:
            wk = (t >= holdout["start_week"]) & (t <= holdout["end_week"])
            geo_spend.loc[wk, holdout["channel"]] = 0.0
        contrib = {}
        for c in CHANNELS:
            x = geo_spend[c].to_numpy() / share[g] / scale[c]
            contrib[c] = share[g] * p.beta[c] * logistic_saturation(
                geometric_adstock(x, p.alpha[c]), p.lam[c]
            )
        base = share[g] * (
            p.intercept + p.trend * t + p.season_amp * season + p.promo_effect * promo + shock
        )
        noise = share[g] * rng.normal(0, p.geo_noise_sd, p.n_weeks)
        sales = base + sum(contrib.values()) + noise
        df = pd.DataFrame(
            {
                "date": dates,
                "week": t,
                "geo": g,
                "pop_share": share[g],
                "sales": sales,
                "promo": promo,
                **{f"{c}_spend": geo_spend[c].to_numpy() for c in CHANNELS},
                **{f"{c}_true_contrib": contrib[c] for c in CHANNELS},
            }
        )
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)

    agg_cols = ["sales"] + [f"{c}_spend" for c in CHANNELS] + [f"{c}_true_contrib" for c in CHANNELS]
    national = panel.groupby(["date", "week"], as_index=False)[agg_cols].sum()
    national["sales"] += rng.normal(0, p.noise_sd, p.n_weeks)
    national["promo"] = promo
    national["trend"] = t / p.n_weeks
    national = national.rename(columns={f"{c}_spend": c for c in CHANNELS})

    truth = pd.DataFrame(
        {
            "channel": CHANNELS,
            "spend": [national[c].sum() for c in CHANNELS],
            "contribution": [national[f"{c}_true_contrib"].sum() for c in CHANNELS],
            "alpha": [p.alpha[c] for c in CHANNELS],
            "lam": [p.lam[c] for c in CHANNELS],
        }
    )
    truth["roi"] = truth["contribution"] / truth["spend"]
    return {"panel": panel, "national": national, "truth": truth, "params": p, "shock": shock}


def default_holdout(n_geos=40, n_treated=10, seed=11):
    """Search holdout: 10 randomly chosen geos turn off search for 7 weeks near the end of the sample."""
    rng = np.random.default_rng(seed)
    geos = sorted(rng.choice(n_geos, size=n_treated, replace=False).tolist())
    return {"channel": "search", "geos": geos, "start_week": 130, "end_week": 136}
