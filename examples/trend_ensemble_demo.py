"""Visualize the cleaned trend ensemble on BTC-USD with fixed parameters.

This example is descriptive: it does not optimize on BTC-USD before plotting.
The original research used separate parameter and weight searches; see
``TrendEnsemble.optimize_parameters`` and ``TrendEnsemble.optimize_weights``.
"""

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


OUTPUT_DIR = Path(__file__).parent / "output"


def download_btc() -> pd.DataFrame:
    raw = yf.download("BTC-USD", period="5y", interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No BTC-USD data returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = raw[["Open", "High", "Low", "Close"]].rename(columns=str.lower).dropna().copy()
    data["hl2"] = (data["high"] + data["low"]) / 2
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = download_btc()

    ensemble = TrendEnsemble(DEFAULT_MODELS)
    components = ensemble.component_signals(data).fillna(0)
    combined = ensemble.signal(data).fillna(0)

    changes = combined.ne(combined.shift(1)) & combined.ne(0)
    long_changes = changes & combined.eq(1)
    short_changes = changes & combined.eq(-1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data["close"], linewidth=1.4, label="BTC-USD close")
    ax.scatter(data.index[long_changes], data.loc[long_changes, "close"], marker="^", s=36, label="Ensemble turns long")
    ax.scatter(data.index[short_changes], data.loc[short_changes, "close"], marker="v", s=36, label="Ensemble turns short")
    ax.set_title("BTC-USD with fixed-parameter trend-ensemble state changes")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trend_ensemble_signal.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    values = components.T.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(components.columns)))
    ax.set_yticklabels(components.columns)
    tick_positions = np.linspace(0, len(data) - 1, 6, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([data.index[i].strftime("%Y-%m") for i in tick_positions])
    ax.set_title("Trend-ensemble component votes (-1 / 0 / +1)")
    fig.colorbar(image, ax=ax, label="Directional vote")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trend_ensemble_components.png", dpi=160)
    plt.close(fig)

    print(f"Rows: {len(data):,}")
    print(f"Models: {len(components.columns)}")
    print(f"State changes: {int(changes.sum())}")
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
