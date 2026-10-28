"""User-level journeys with known incremental effects, and attribution rules.

Purchase intent is latent. High-intent users are more likely to search (and to
open emails) and more likely to buy regardless of ads, so any method that only
looks at who converted after which touch will credit search with sales it did
not cause. The simulator knows each user's conversion probability with and
without each channel, so true incremental conversions can be computed exactly.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

CHANNELS = ["display", "social", "email", "search"]

TRUE_EFFECT = {"display": 0.25, "social": 0.35, "email": 0.30, "search": 0.50}


def simulate_journeys(n_users=50_000, seed=3):
    rng = np.random.default_rng(seed)
    high = rng.random(n_users) < 0.2
    expo = pd.DataFrame(
        {
            "display": rng.random(n_users) < 0.50,
            "social": rng.random(n_users) < 0.35,
            "email": rng.random(n_users) < np.where(high, 0.50, 0.25),
            "search": rng.random(n_users) < np.where(high, 0.70, 0.15),
        }
    )

    def p_convert(exposure):
        z = -3.2 + 1.8 * high + sum(TRUE_EFFECT[c] * exposure[c].to_numpy() for c in CHANNELS)
        return 1 / (1 + np.exp(-z))

    p = p_convert(expo)
    converted = rng.random(n_users) < p

    true_incr = {}
    for c in CHANNELS:
        cf = expo.copy()
        cf[c] = False
        true_incr[c] = float((p - p_convert(cf)).sum())

    position = {"display": 0.0, "social": 0.5, "email": 1.0, "search": 2.0}
    paths = []
    for i in range(n_users):
        touched = [c for c in CHANNELS if expo.at[i, c]]
        order = sorted(touched, key=lambda c: position[c] + rng.normal(0, 0.6))
        paths.append(order)

    users = expo.assign(high_intent=high, converted=converted, path=paths)
    return users, pd.Series(true_incr, name="true_incremental")


def rule_based(users):
    """First-touch, last-touch and linear credit among converting users."""
    conv = users[users["converted"] & (users["path"].str.len() > 0)]
    first = conv["path"].str[0].value_counts()
    last = conv["path"].str[-1].value_counts()
    linear = pd.Series(0.0, index=CHANNELS)
    for path in conv["path"]:
        for c in path:
            linear[c] += 1 / len(path)
    return pd.DataFrame({"first_touch": first, "last_touch": last, "linear": linear}).reindex(CHANNELS).fillna(0)


def _markov_conversion_prob(paths, converted, removed=None):
    """First-order Markov chain over states start -> channels -> conv/null."""
    states = ["start"] + CHANNELS + ["conv", "null"]
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    for path, y in zip(paths, converted):
        seq = ["start"] + list(path) + (["conv"] if y else ["null"])
        for a, b in zip(seq[:-1], seq[1:]):
            counts[idx[a], idx[b]] += 1
    T = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    T[idx["conv"], idx["conv"]] = 1
    T[idx["null"], idx["null"]] = 1
    if removed is not None:
        r = idx[removed]
        T[:, r] = 0
        T[:, idx["null"]] += 1 - T.sum(1)
    transient = [idx[s] for s in ["start"] + CHANNELS]
    Q = T[np.ix_(transient, transient)]
    R = T[np.ix_(transient, [idx["conv"]])]
    N = np.linalg.solve(np.eye(len(transient)) - Q, R)
    return N[0, 0]


def markov(users):
    paths, y = users["path"].tolist(), users["converted"].to_numpy()
    base = _markov_conversion_prob(paths, y)
    removal = pd.Series({c: 1 - _markov_conversion_prob(paths, y, removed=c) / base for c in CHANNELS})
    total = users.loc[users["path"].str.len() > 0, "converted"].sum()
    return (removal / removal.sum() * total).rename("markov")


def shapley(users):
    """Shapley values where a coalition's value is the conversions from users
    whose set of touched channels is contained in the coalition."""
    sets = users["path"].apply(frozenset)
    conv_by_set = users.groupby(sets)["converted"].sum()

    def v(coalition):
        s = frozenset(coalition)
        return sum(n for k, n in conv_by_set.items() if k and k <= s)

    n = len(CHANNELS)
    out = {}
    for c in CHANNELS:
        others = [o for o in CHANNELS if o != c]
        val = 0.0
        for k in range(n):
            for S in combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                val += w * (v(S + (c,)) - v(S))
        out[c] = val
    return pd.Series(out, name="shapley")
