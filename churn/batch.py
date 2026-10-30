"""Many customers at once: top risk factors and rule matching without a Python loop per rule and row.

The single-customer versions live in explain.py (top_drivers) and recommend.py (recommend). The functions
here must give the same answers; tests/test_workspace.py compares them row by row."""
from typing import List, Sequence

import numpy as np
import pandas as pd

from .recommend import OPS, Rule


def top_factor_mask(contrib: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """True where a column is among the row's k strongest risk-RAISING contributions."""
    ranks = contrib.where(contrib > 0).rank(axis=1, ascending=False, method="first")
    return ranks <= k


def top_factor_names(contrib: pd.DataFrame, k: int = 3) -> List[List[str]]:
    """For each row, the names of up to k risk-raising columns, strongest first (empty list if none raise risk)."""
    if contrib.empty:
        return []
    vals = contrib.to_numpy(dtype=float)
    cols = np.asarray(contrib.columns, dtype=object)
    order = np.argsort(-np.where(vals > 0, vals, -np.inf), axis=1, kind="stable")[:, :k]
    out = []
    for r in range(vals.shape[0]):
        out.append([str(cols[j]) for j in order[r] if vals[r, j] > 0])
    return out


def _condition_mask(cond: dict, raw: pd.DataFrame) -> np.ndarray:
    n = len(raw)
    feature = cond.get("feature")
    if feature not in raw.columns:
        return np.zeros(n, dtype=bool)
    col = raw[feature]
    present = col.notna().to_numpy()
    op, value = cond.get("op"), cond.get("value")
    try:
        if op == "in":
            options = list(value)
            hit = col.isin(options).to_numpy() | col.astype(str).isin([str(x) for x in options]).to_numpy()
        else:
            hit = np.asarray(OPS[op](col, value), dtype=bool)
    except (TypeError, KeyError, ValueError):
        return np.zeros(n, dtype=bool)
    return hit & present


def recommend_batch(rules: Sequence[Rule], risk_levels, contrib: pd.DataFrame, raw: pd.DataFrame,
                    max_actions: int = 3, k_drivers: int = 5, k_factors: int = 3) -> pd.DataFrame:
    """Rule matching for many customers. `contrib` and `raw` must hold the same customers in the same order.

    Same semantics as recommend.recommend(): a rule applies when the risk level is allowed, its driver is among
    the customer's k strongest risk-raising factors, and every condition holds. Rules are ordered by priority
    (higher first), then by how strongly the driver raises this customer's risk, then by file order.
    Returns one row per customer with: top_factors, n_matched, action_ids, actions, next_action."""
    n = len(raw)
    if len(contrib) != n:
        raise ValueError("contrib and raw must describe the same customers.")
    levels = np.asarray(risk_levels, dtype=object)
    top = top_factor_mask(contrib, k_drivers)
    matches = np.zeros((n, len(rules)), dtype=bool)
    strength = np.zeros((n, len(rules)), dtype=float)
    for j, r in enumerate(rules):
        ok = np.ones(n, dtype=bool)
        if r.risk_levels is not None:
            ok &= np.isin(levels, list(r.risk_levels))
        if r.driver is not None:
            if r.driver in top.columns:
                ok &= top[r.driver].to_numpy()
                strength[:, j] = contrib[r.driver].to_numpy(dtype=float)
            else:
                ok &= False
        for cond in r.conditions:
            ok &= _condition_mask(cond, raw)
        matches[:, j] = ok
    priority = np.array([r.priority for r in rules], dtype=float)
    ids = [r.id for r in rules]
    texts = [r.action for r in rules]
    action_ids, action_texts, counts = [], [], []
    for i in range(n):
        hit = np.flatnonzero(matches[i])
        counts.append(int(hit.size))
        if hit.size == 0:
            action_ids.append([])
            action_texts.append([])
            continue
        order = sorted(hit, key=lambda j: (-priority[j], -strength[i, j], j))[:max_actions]
        action_ids.append([ids[j] for j in order])
        action_texts.append([texts[j] for j in order])
    return pd.DataFrame({
        "top_factors": top_factor_names(contrib, k_factors),
        "n_matched": counts,
        "action_ids": action_ids,
        "actions": action_texts,
        "next_action": [a[0] if a else "" for a in action_texts],
    }, index=raw.index)
