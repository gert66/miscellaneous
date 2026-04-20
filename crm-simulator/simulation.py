"""
Core Monte Carlo simulation logic for the Time-to-CRM simulator.
No framework dependencies — pure Python + NumPy.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class SimulationParams:
    # Volume
    num_leads: int = 500

    # Conversion rates (0–1)
    lead_to_mql_rate: float = 0.40
    mql_to_sql_rate: float = 0.35
    sql_to_opp_rate: float = 0.60
    opp_win_rate: float = 0.25

    # Stage durations: mean days
    days_lead_to_mql_mean: float = 7.0
    days_mql_to_sql_mean: float = 14.0
    days_sql_to_opp_mean: float = 10.0
    days_opp_to_close_mean: float = 45.0

    # Stage durations: std-dev days (variability)
    days_lead_to_mql_std: float = 3.0
    days_mql_to_sql_std: float = 7.0
    days_sql_to_opp_std: float = 5.0
    days_opp_to_close_std: float = 20.0

    # Random seed (-1 = random)
    seed: int = -1


def _sample_duration(rng: np.random.Generator, mean: float, std: float, n: int) -> np.ndarray:
    """Sample non-negative stage durations from a log-normal distribution."""
    if std <= 0:
        return np.full(n, mean)
    # Convert normal (mean, std) to log-normal parameters
    variance = std ** 2
    sigma2 = np.log(1 + variance / mean ** 2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), n)


def _percentile_stats(arr: np.ndarray) -> Dict:
    if len(arr) == 0:
        return {"mean": 0, "median": 0, "p10": 0, "p25": 0, "p75": 0, "p90": 0, "min": 0, "max": 0}
    return {
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "p10": round(float(np.percentile(arr, 10)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "min": round(float(np.min(arr)), 1),
        "max": round(float(np.max(arr)), 1),
    }


def _histogram(arr: np.ndarray, bins: int = 40) -> Dict:
    if len(arr) == 0:
        return {"labels": [], "values": []}
    counts, edges = np.histogram(arr, bins=bins)
    labels = [round((edges[i] + edges[i + 1]) / 2, 1) for i in range(len(counts))]
    return {"labels": labels, "values": counts.tolist()}


def _cdf(arr: np.ndarray, points: int = 100) -> Dict:
    if len(arr) == 0:
        return {"x": [], "y": []}
    sorted_arr = np.sort(arr)
    y = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    idx = np.linspace(0, len(sorted_arr) - 1, min(points, len(sorted_arr))).astype(int)
    return {"x": sorted_arr[idx].tolist(), "y": (y[idx] * 100).tolist()}


def _stage_boxplot(arr: np.ndarray) -> Dict:
    if len(arr) == 0:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
    return {
        "min": round(float(np.percentile(arr, 5)), 1),
        "q1": round(float(np.percentile(arr, 25)), 1),
        "median": round(float(np.median(arr)), 1),
        "q3": round(float(np.percentile(arr, 75)), 1),
        "max": round(float(np.percentile(arr, 95)), 1),
    }


def run_simulation(params: SimulationParams) -> Dict:
    """
    Run a Monte Carlo sales-pipeline simulation.

    Returns a dict with funnel counts, timing distributions, and summary stats
    suitable for direct JSON serialisation.
    """
    seed = None if params.seed < 0 else params.seed
    rng = np.random.default_rng(seed)

    n = params.num_leads

    # ── Stage conversion masks ────────────────────────────────────────────────
    mql_mask = rng.random(n) < params.lead_to_mql_rate
    sql_mask = mql_mask & (rng.random(n) < params.mql_to_sql_rate)
    opp_mask = sql_mask & (rng.random(n) < params.sql_to_opp_rate)
    won_mask = opp_mask & (rng.random(n) < params.opp_win_rate)

    # ── Stage durations (sampled for all leads; only meaningful for converters) ─
    t1 = _sample_duration(rng, params.days_lead_to_mql_mean, params.days_lead_to_mql_std, n)
    t2 = _sample_duration(rng, params.days_mql_to_sql_mean, params.days_mql_to_sql_std, n)
    t3 = _sample_duration(rng, params.days_sql_to_opp_mean, params.days_sql_to_opp_std, n)
    t4 = _sample_duration(rng, params.days_opp_to_close_mean, params.days_opp_to_close_std, n)

    # ── Cumulative times for leads that reach each stage ─────────────────────
    time_to_mql = t1[mql_mask]
    time_to_sql = (t1 + t2)[sql_mask]
    time_to_opp = (t1 + t2 + t3)[opp_mask]   # ← "Time to CRM entry"
    time_to_close = (t1 + t2 + t3 + t4)[won_mask]

    # ── Funnel counts ─────────────────────────────────────────────────────────
    funnel = {
        "Lead": n,
        "MQL": int(mql_mask.sum()),
        "SQL": int(sql_mask.sum()),
        "Opportunity": int(opp_mask.sum()),
        "Won": int(won_mask.sum()),
    }

    # ── Conversion rates ─────────────────────────────────────────────────────
    conversion_rates = {
        "Lead→MQL": round(funnel["MQL"] / n * 100, 1) if n else 0,
        "MQL→SQL": round(funnel["SQL"] / funnel["MQL"] * 100, 1) if funnel["MQL"] else 0,
        "SQL→Opp": round(funnel["Opportunity"] / funnel["SQL"] * 100, 1) if funnel["SQL"] else 0,
        "Opp→Won": round(funnel["Won"] / funnel["Opportunity"] * 100, 1) if funnel["Opportunity"] else 0,
        "Overall": round(funnel["Won"] / n * 100, 2) if n else 0,
    }

    # ── Per-stage duration distributions (only converters) ───────────────────
    stage_durations = {
        "Lead→MQL": _stage_boxplot(t1[mql_mask]),
        "MQL→SQL": _stage_boxplot(t2[sql_mask]),
        "SQL→Opp": _stage_boxplot(t3[opp_mask]),
        "Opp→Close": _stage_boxplot(t4[won_mask]),
    }

    return {
        "funnel": funnel,
        "conversion_rates": conversion_rates,
        "stage_durations": stage_durations,
        "time_to_crm": {
            "stats": _percentile_stats(time_to_opp),
            "histogram": _histogram(time_to_opp),
            "cdf": _cdf(time_to_opp),
            "count": len(time_to_opp),
        },
        "time_to_close": {
            "stats": _percentile_stats(time_to_close),
            "histogram": _histogram(time_to_close),
            "count": len(time_to_close),
        },
        "time_to_mql": {
            "stats": _percentile_stats(time_to_mql),
        },
        "time_to_sql": {
            "stats": _percentile_stats(time_to_sql),
        },
    }


def params_from_dict(d: Dict) -> SimulationParams:
    """Parse and validate a JSON request body into SimulationParams."""
    def pct(key, default):
        return max(0.0, min(1.0, float(d.get(key, default)) / 100.0))

    def pos(key, default, lo=0.1):
        return max(lo, float(d.get(key, default)))

    return SimulationParams(
        num_leads=max(10, min(10_000, int(d.get("num_leads", 500)))),
        lead_to_mql_rate=pct("lead_to_mql_rate", 40),
        mql_to_sql_rate=pct("mql_to_sql_rate", 35),
        sql_to_opp_rate=pct("sql_to_opp_rate", 60),
        opp_win_rate=pct("opp_win_rate", 25),
        days_lead_to_mql_mean=pos("days_lead_to_mql_mean", 7),
        days_mql_to_sql_mean=pos("days_mql_to_sql_mean", 14),
        days_sql_to_opp_mean=pos("days_sql_to_opp_mean", 10),
        days_opp_to_close_mean=pos("days_opp_to_close_mean", 45),
        days_lead_to_mql_std=pos("days_lead_to_mql_std", 3, 0),
        days_mql_to_sql_std=pos("days_mql_to_sql_std", 7, 0),
        days_sql_to_opp_std=pos("days_sql_to_opp_std", 5, 0),
        days_opp_to_close_std=pos("days_opp_to_close_std", 20, 0),
        seed=int(d.get("seed", -1)),
    )
