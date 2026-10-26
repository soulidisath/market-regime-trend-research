"""Reproducible BTC-USD market-regime example.

Downloads daily BTC-USD OHLC data with yfinance, computes the cleaned Market
Regime Indicator, and writes two figures to examples/output/:

1. BTC price with periods classified as trending shaded in the background.
2. The five component regime votes (ADF, KPSS, Hurst, ADX, Phillips-Perron).

The figures are descriptive diagnostics for the research code. They are not a
trading-performance claim.
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


TICKER = "BTC-USD"
START = "2021-01-01"
END = "2026-01-01"

# Representative parameters from the ranges explored in the original research.
# They are fixed here for reproducibility; this example does not re-optimize them.
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
    """Download and normalize daily BTC-USD OHLC data."""
    data = yf.download(
        TICKER,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data returned by yfinance.")

    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.lower)
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    return data[required].dropna().copy()


def contiguous_regions(mask: pd.Series):
    """Yield (start, end) index pairs for contiguous True regions."""
    values = mask.fillna(False).to_numpy(dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts, ends):
        yield mask.index[start], mask.index[end]


def plot_regimes(data: pd.DataFrame, mri: pd.Series, output_dir: Path) -> None:
    """Plot price with trending periods shaded."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(data.index, data["close"], linewidth=1.2, label="BTC-USD close")

    trending = mri > 0.5
    first = True
    for start, end in contiguous_regions(trending):
        ax.axvspan(
            start,
            end,
            alpha=0.15,
            label="MRI > 0.5 (trending vote)" if first else None,
        )
        first = False

    ax.set_yscale("log")
    ax.set_title("BTC-USD with Market Regime Indicator classification")
    ax.set_ylabel("Price (USD, log scale)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "btc_regimes.png", dpi=160)
    plt.close(fig)


def plot_components(indicators: dict[str, pd.Series], output_dir: Path) -> None:
    """Plot the binary component votes as a compact heatmap."""
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
    ax.set_title("Component regime votes (0 = non-trending vote, 1 = trending vote)")
    fig.colorbar(image, ax=ax, ticks=[0, 1], fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "btc_regime_components.png", dpi=160)
    plt.close(fig)


def main() -> None:
    data = load_ohlc()
    model = MarketRegimeIndicator(MRI_PARAMS)
    mri, indicators, score = model.calculate(data)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_regimes(data, mri, output_dir)
    plot_components(indicators, output_dir)

    valid = mri.dropna()
    trending_share = float((valid > 0.5).mean()) if not valid.empty else float("nan")

    print(f"Rows: {len(data):,}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Heuristic regime score: {score:.4f}")
    print(f"Share classified as trending: {trending_share:.1%}")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
