"""Visualize the cleaned trend ensemble on WTI crude futures with fixed parameters.

The example uses the ensemble's default indicator parameters and equal weights.
It does not optimize on WTI immediately before plotting. The two figures are
intended to make the voting mechanism easy to understand, not to claim trading
performance.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_ensemble import DEFAULT_MODELS, TrendEnsemble  # noqa: E402

TICKER = "CL=F"
START = "2021-01-01"
END = "2026-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_ohlc() -> pd.DataFrame:
    raw = yf.download(
        TICKER,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("No CL=F data returned by yfinance.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    data = raw[["Open", "High", "Low", "Close"]].rename(columns=str.lower).dropna().copy()
    data["hl2"] = (data["high"] + data["low"]) / 2
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_ohlc()

    ensemble = TrendEnsemble(DEFAULT_MODELS)
    components = ensemble.component_signals(data).fillna(0.0)
    combined = ensemble.signal(data).fillna(0.0)

    nonzero = components.ne(0).sum(axis=1).replace(0, np.nan)
    bullish_share = components.gt(0).sum(axis=1).div(nonzero).mul(100.0)

    changes = combined.ne(combined.shift(1)) & combined.ne(0)
    long_changes = changes & combined.eq(1)
    short_changes = changes & combined.eq(-1)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(data.index, data["close"], linewidth=1.25, label="WTI crude futures close")
    ax.scatter(
        data.index[long_changes],
        data.loc[long_changes, "close"],
        marker="^",
        s=38,
        label="Ensemble turns bullish",
    )
    ax.scatter(
        data.index[short_changes],
        data.loc[short_changes, "close"],
        marker="v",
        s=38,
        label="Ensemble turns bearish",
    )
    ax.set_title("WTI crude futures: fixed-parameter trend-ensemble state changes")
    ax.set_ylabel("Futures price")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wti_trend_ensemble_signal.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(bullish_share.index, bullish_share, linewidth=1.35)
    ax.axhline(50.0, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_title("WTI trend ensemble: share of active models voting bullish")
    ax.set_ylabel("Bullish vote share (%)")
    ax.set_xlabel("")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wti_trend_vote_share.png", dpi=160)
    plt.close(fig)

    current_share = float(bullish_share.dropna().iloc[-1]) if bullish_share.notna().any() else float("nan")
    print(f"Rows: {len(data):,}")
    print(f"Models: {len(components.columns)}")
    print(f"State changes: {int(changes.sum())}")
    print(f"Latest bullish vote share: {current_share:.1f}%")
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
