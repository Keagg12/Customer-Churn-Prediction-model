"""Rule-based retention recommendations. Rules live in config/rules.json and are written by the
project owner against real feature names; nothing here assumes any particular column."""
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "==": operator.eq, "!=": operator.ne}
LEVELS = ("Low", "Medium", "High")


@dataclass
class Rule:
    id: str
    action: str
    risk_levels: Optional[List[str]] = None   # None = any level
    driver: Optional[str] = None               # feature that must be among the top risk-raising drivers
    conditions: List[dict] = field(default_factory=list)  # [{"feature":..,"op":..,"value":..}]
    priority: int = 0


def parse_rules(data: dict) -> List[Rule]:
    rules = []
    for r in data.get("rules", []):
        rules.append(Rule(id=str(r.get("id", f"rule{len(rules) + 1}")), action=r["action"],
                          risk_levels=r.get("risk_levels"), driver=r.get("driver"),
                          conditions=r.get("conditions", []), priority=int(r.get("priority", 0))))
    return rules


def load_rules(path) -> List[Rule]:
    return parse_rules(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_rules(rules: List[Rule], known_features) -> List[str]:
    known = set(known_features)
    problems = []
    for r in rules:
        if not r.action or not str(r.action).strip():
            problems.append(f"Rule '{r.id}': action text is empty.")
        if r.risk_levels is not None:
            bad = [x for x in r.risk_levels if x not in LEVELS]
            if bad:
                problems.append(f"Rule '{r.id}': unknown risk level(s) {bad}; use {list(LEVELS)}.")
        if r.driver is not None and r.driver not in known:
            problems.append(f"Rule '{r.id}': driver '{r.driver}' is not a feature column.")
        for c in r.conditions:
            if c.get("feature") not in known:
                problems.append(f"Rule '{r.id}': condition feature '{c.get('feature')}' is not a feature column.")
            if c.get("op") not in OPS and c.get("op") != "in":
                problems.append(f"Rule '{r.id}': unknown operator '{c.get('op')}'.")
    ids = [r.id for r in rules]
    problems += [f"Duplicate rule id '{i}'." for i in set(ids) if ids.count(i) > 1]
    return problems


def _holds(cond: dict, row: pd.Series) -> bool:
    v = row.get(cond["feature"])
    if v is None or pd.isna(v):
        return False
    try:
        if cond["op"] == "in":
            return v in cond["value"] or str(v) in [str(x) for x in cond["value"]]
        return bool(OPS[cond["op"]](v, cond["value"]))
    except TypeError:
        return False


def recommend(rules: List[Rule], risk_level: str, drivers: List[tuple], row: pd.Series, max_actions: int = 3) -> Dict:
    """drivers = [(feature, contribution)] for risk-raising factors, strongest first.
    At most `max_actions` actions are returned (highest priority, then strongest driver)."""
    strength = dict(drivers)
    matched = []
    for r in rules:
        if r.risk_levels is not None and risk_level not in r.risk_levels:
            continue
        if r.driver is not None and r.driver not in strength:
            continue
        if not all(_holds(c, row) for c in r.conditions):
            continue
        matched.append((r.priority, strength.get(r.driver, 0.0), r))
    matched.sort(key=lambda t: (-t[0], -t[1]))
    n_matched = len(matched)
    matched = matched[:max_actions]
    actions = []
    for _, _, r in matched:
        because = []
        if r.driver:
            because.append(f"'{r.driver}' is one of the main factors raising this customer's risk")
        because += [f"{c['feature']} {c['op']} {c['value']}" for c in r.conditions]
        actions.append({"rule_id": r.id, "action": r.action, "because": "; ".join(because) or f"risk level is {risk_level}"})
    names = ", ".join(f for f, _ in drivers[:3]) or "none identified"
    if actions:
        more = f" ({n_matched - len(actions)} more rule(s) also matched)" if n_matched > len(actions) else ""
        summary = f"{len(actions)} recommended action(s){more}. Main risk-raising factors: {names}."
    else:
        summary = f"No configured rule matched. Main risk-raising factors: {names}."
    return {"risk_level": risk_level, "actions": actions, "matched": bool(actions), "summary": summary}
