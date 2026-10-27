"""Reproducible E-mini S&P 500 futures market-regime example.

Downloads daily ES=F OHLC data with yfinance, applies a fixed Market Regime
Indicator parameter set, and writes two descriptive diagnostics to
``examples/output/``.

The five component models each cast a binary vote. The aggregate MRI is their
mean, so an MRI above 0.5 means that at least three of the five components vote
for a trending regime. The example does not optimize parameters immediately
before displaying the same period.
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

from src.market_regime import MarketRegimeIndicator  # noqa: E402

TICKER = "ES=F"
START = "2021-01-01"
END = "2026-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MRI_PARAMS = {
    "adf_length": 100,
    "adf_threshold": -1.5,
    "kpss_length": 100,
    "kpss_threshold": 1.0,
    "hurst_length": 70,
    "hurst_threshold": 0.50,
    "adx_length": 20,
    "adx_threshold": 22.0,
    "pp_length": 100,
    "pp_threshold": -1.75,
}


def load_ohlc() -> pd.DataFrame:
    data = yf.download(
        TICKER,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("No ES=F data returned by yfinance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.lower)
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")
    return data[required].dropna().copy()


def contiguous_regions(mask: pd.Series):
    values = mask.fillna(False).to_numpy(dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts, ends):
        yield mask.index[start], mask.index[end]


def plot_regimes(data: pd.DataFrame, mri: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(data.index, data["close"], linewidth=1.25, label="E-mini S&P 500 futures close")

    trending = mri > 0.5
    first = True
    for start, end in contiguous_regions(trending):
        ax.axvspan(
            start,
            end,
            alpha=0.15,
            label="At least 3 of 5 components vote trending" if first else None,
        )
        first = False

    ax.set_title("E-mini S&P 500 futures: market-regime classification")
    ax.set_ylabel("Futures price")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sp500_futures_regimes.png", dpi=160)
    plt.close(fig)


def plot_components(indicators: dict[str, pd.Series]) -> None:
    frame = pd.DataFrame(indicators).dropna(how="all")
    if frame.empty:
        raise RuntimeError("No indicator values were produced.")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    image = ax.imshow(
        frame.T.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    tick_count = min(8, len(frame))
    ticks = np.linspace(0, len(frame) - 1, tick_count, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([frame.index[i].strftime("%Y-%m") for i in ticks])
    ax.set_yticks(range(len(frame.columns)))
    ax.set_yticklabels([name.upper() for name in frame.columns])
    ax.set_title("Five regime votes (0 = non-trending, 1 = trending)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sp500_futures_regime_components.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_ohlc()
    model = MarketRegimeIndicator(MRI_PARAMS)

    result = model.calculate(data)
    mri = result[0]
    indicators = result[1]
    score = result[2]

    plot_regimes(data, mri)
    plot_components(indicators)

    valid = mri.dropna()
    trending_share = float((valid > 0.5).mean()) if not valid.empty else float("nan")
    average_vote_share = float(valid.mean()) if not valid.empty else float("nan")

    print(f"Rows: {len(data):,}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Average trending-vote share: {average_vote_share:.1%}")
    print(f"Share of observations with >=3/5 trending votes: {trending_share:.1%}")
    print(f"Legacy heuristic comparison score: {score:.4f}")
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
