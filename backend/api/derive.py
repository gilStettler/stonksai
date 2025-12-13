from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# -----------------------------
# EWMA helpers
# -----------------------------
def calculate_ewma_volatility(log_returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    EWMA volatility of log returns, standard finance EWMA.
    """
    r = log_returns.fillna(0.0).to_numpy(dtype=float)
    if len(r) == 0:
        return pd.Series([], index=log_returns.index)

    ewma_var = np.zeros(len(r), dtype=float)
    ewma_var[0] = r[0] ** 2
    for t in range(1, len(r)):
        ewma_var[t] = lambda_ * ewma_var[t - 1] + (1.0 - lambda_) * (r[t] ** 2)

    return pd.Series(np.sqrt(ewma_var), index=log_returns.index)


def compute_realized_ewma_from_csv(csv_path: Path, lambda_: float = 0.94) -> pd.DataFrame:
    """
    Reads stock CSV and computes realized EWMA vol from Close prices.
    Expects:
      - timestamp column (date index)
      - Close column
    Returns DataFrame with columns: timestamp, realized_ewma
    """
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing 'timestamp': {csv_path}")
    if "Close" not in df.columns and "close" not in df.columns:
        raise ValueError(f"CSV missing 'Close'/'close': {csv_path}")

    close_col = "Close" if "Close" in df.columns else "close"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # log returns
    df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))
    df["realized_ewma"] = calculate_ewma_volatility(df["log_return"], lambda_=lambda_)

    return df[["timestamp", "realized_ewma"]].dropna()


def last_n_realized(csv_latest_dir: Path, symbol: str, n: int = 5, lambda_: float = 0.94) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """
    Returns:
      - list of last n realized points [{date, realized}]
      - last_realized value (float) or None
    """
    csv_path = csv_latest_dir / f"data_{symbol}.csv"
    if not csv_path.exists():
        return [], None

    realized_df = compute_realized_ewma_from_csv(csv_path, lambda_=lambda_)
    if realized_df.empty:
        return [], None

    tail = realized_df.tail(n).copy()
    points = []
    for _, row in tail.iterrows():
        points.append({
            "date": row["timestamp"].date().isoformat(),
            "realized": float(row["realized_ewma"]),
        })

    last_val = float(tail["realized_ewma"].iloc[-1])
    return points, last_val


def range_from_points(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [p["realized"] for p in points if p.get("realized") is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "pos": None}

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return {"n": len(vals), "min": vmin, "max": vmax, "pos": None}


def position_in_range(value: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> Optional[float]:
    if value is None or vmin is None or vmax is None:
        return None
    if vmax == vmin:
        return 0.5
    return float((value - vmin) / (vmax - vmin))


def delta_arrow(delta: Optional[float], eps: float = 1e-6) -> Optional[str]:
    if delta is None:
        return None
    if delta > eps:
        return "↑"
    if delta < -eps:
        return "↓"
    return "→"


def risk_from_pos(pos: Optional[float]) -> Dict[str, Any]:
    """
    Simple beginner-friendly risk label based on where forecast lies inside last-n realized range.
    """
    if pos is None:
        return {"label": "Medium", "reason": "missing_pos"}

    if pos <= 0.33:
        return {"label": "Low", "reason": "forecast_near_recent_low"}
    if pos >= 0.66:
        return {"label": "High", "reason": "forecast_near_recent_high"}
    return {"label": "Medium", "reason": "forecast_mid_range"}


# -----------------------------
# Public derive function
# -----------------------------
def derive_metrics(
    symbol: str,
    history_records: List[Dict[str, Any]],
    csv_latest_dir: Path,
    ewma_lambda: float = 0.94,
    range_days: int = 5,
) -> Dict[str, Any]:
    """
    Produces:
      - confidence_band from latest record
      - 5d realized range from CSV (min/max/n + pos)
      - delta_vol comparing forecast vs last realized from CSV
      - risk label based on pos
    """
    if not history_records:
        return {}

    latest = history_records[-1]
    try:
        forecast_val = float(latest.get("forecast_value"))
    except Exception:
        forecast_val = None

    # Confidence band from latest record
    band = {}
    try:
        band = {
            "low": float(latest.get("confidence_low")),
            "high": float(latest.get("confidence_high")),
        }
    except Exception:
        band = {"low": None, "high": None}

    # CSV-based realized points
    points, last_realized = last_n_realized(csv_latest_dir, symbol, n=range_days, lambda_=ewma_lambda)
    r = range_from_points(points)

    # pos of forecast within realized range
    pos = position_in_range(forecast_val, r["min"], r["max"])
    r["pos"] = pos
    r["points"] = points  # useful for UI mini chart

    # delta (forecast vs last realized)
    dvol = None
    if forecast_val is not None and last_realized is not None:
        dvol = float(forecast_val - last_realized)

    out = {
        "confidence_band": band,
        "range_5d": r,  # still called 5d, but includes n + points
        "delta_vol": dvol,
        "delta_arrow": delta_arrow(dvol),
        "risk": risk_from_pos(pos),
        "last_realized": last_realized,
    }
    return out
