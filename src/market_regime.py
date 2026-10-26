"""Market-regime classification experiments.

The MarketRegimeIndicator combines five binary signals:
ADF, KPSS, Phillips-Perron, a rolling Hurst estimate, and ADX. Their average is
used as a simple regime score, where values above 0.5 are treated as trending.

This code was developed as exploratory research. The custom scoring function is
a heuristic for comparing parameter sets, not a statistical proof that the
classification is economically optimal.
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import optuna
import pandas as pd
from arch.unitroot import PhillipsPerron
from dask.distributed import Client
from statsmodels.tsa import stattools


def rolling_kpss(close: pd.Series, length: int, threshold: float = 1.0) -> pd.Series:
    def test(window):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, *_ = stattools.kpss(window, regression="c", nlags="auto")
            return stat

    values = close.rolling(length, min_periods=length).apply(test)
    return (values >= threshold).astype(int)


def rolling_adf(
    close: pd.Series, length: int, threshold: float = -1.0, smooth: int = 5
) -> pd.Series:
    def test(window):
        stat, *_ = stattools.adfuller(window)
        return stat

    values = close.rolling(length, min_periods=length).apply(test).rolling(smooth).mean()
    return (values >= threshold).astype(int)


def _hurst_rs(values: np.ndarray) -> float:
    """Estimate the Hurst exponent with a simple rescaled-range regression.

    The calculation is intentionally compact and transparent for the portfolio
    version of the project. It is a descriptive long-memory measure rather than
    a precise structural estimator.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return np.nan

    # Work with changes so the level of the price series does not dominate R/S.
    if np.all(x > 0):
        changes = np.diff(np.log(x))
    else:
        changes = np.diff(x)

    max_size = len(changes) // 2
    sizes = [size for size in (8, 16, 32, 64, 128) if size <= max_size]
    if len(sizes) < 2:
        return np.nan

    rs_points = []
    used_sizes = []
    for size in sizes:
        ratios = []
        for start in range(0, len(changes) - size + 1, size):
            segment = changes[start : start + size]
            std = segment.std(ddof=1)
            if not np.isfinite(std) or std == 0:
                continue
            cumulative = np.cumsum(segment - segment.mean())
            span = cumulative.max() - cumulative.min()
            if span > 0:
                ratios.append(span / std)

        if ratios:
            used_sizes.append(size)
            rs_points.append(float(np.mean(ratios)))

    if len(rs_points) < 2:
        return np.nan

    slope, _ = np.polyfit(np.log(used_sizes), np.log(rs_points), 1)
    return float(slope)


def rolling_hurst(close: pd.Series, length: int, threshold: float = 0.75) -> pd.Series:
    values = close.rolling(length, min_periods=length).apply(_hurst_rs, raw=True)
    return (values >= threshold).astype(int)


def rolling_adx(data: pd.DataFrame, length: int, threshold: float = 20.0) -> pd.Series:
    """Return a binary ADX trend-strength vote using Wilder-style smoothing.

    This implementation keeps the example self-contained and avoids depending on
    a third-party technical-analysis wrapper for a single indicator.
    """
    high = data.high.astype(float)
    low = data.low.astype(float)
    close = data.close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=data.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=data.index,
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / length
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_smoothed = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    minus_smoothed = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    return (adx > threshold).astype(int)


def rolling_pp(
    close: pd.Series, length: int, threshold: float = -2.0, smooth: int = 5
) -> pd.Series:
    def test(window):
        return PhillipsPerron(window).stat

    values = close.rolling(length, min_periods=length).apply(test).rolling(smooth).mean()
    return (values >= threshold).astype(int)


def regime_coherence(mri: pd.Series, indicators: dict[str, pd.Series]) -> float:
    """Heuristic agreement/transition score used during parameter search."""
    mri_values = mri.values
    indicator_values = np.array([series.values for series in indicators.values()])

    penalties = []
    for indicator in indicator_values:
        state_indices = np.where(np.diff(indicator) != 0)[0]
        if len(state_indices) == 0:
            penalties.append(-1.0)
            continue
        state_lengths = np.diff(np.append(state_indices, len(indicator)))
        decay_penalty = np.exp(-0.5 * (state_lengths - 2))
        penalties.append(1 - np.sum(decay_penalty) / len(state_indices))

    transition_coherence = float(np.mean(penalties))
    state_coherence = float(np.mean(indicator_values == mri_values, axis=1).mean() * 1.5)
    state_changes = np.sum(np.diff(indicator_values, axis=1) != 0, axis=1)
    transition_balance = float(1 - 0.005 * np.std(state_changes))
    return (transition_coherence + state_coherence + transition_balance) / 3


def evaluate_regime_score(
    close: pd.Series, mri: pd.Series, indicators: dict[str, pd.Series] | None = None
) -> float:
    """Score how closely labelled segments resemble the intended regimes.

    Mean-reverting segments are rewarded for staying near their segment mean;
    trending segments are rewarded for larger endpoint moves relative to the
    asset's full-sample standard deviation. This was designed for model comparison
    inside the research workflow, not as an investment-performance metric.
    """
    mean_reverting = (mri <= 0.5).astype(int)
    trending = (mri > 0.5).astype(int)

    neutral_segments = (mean_reverting != mean_reverting.shift()).cumsum() * mean_reverting
    segment_lengths = neutral_segments.groupby(neutral_segments).transform("size")
    long_neutral = neutral_segments[segment_lengths >= 10]

    segment_means = close.groupby(long_neutral).transform("mean")
    absolute_deviations = np.abs(close - segment_means)
    mean_deviation = absolute_deviations.groupby(long_neutral).mean().mean()

    asset_std = float(close.std())
    if not np.isfinite(asset_std) or asset_std == 0:
        return 0.0

    neutral_score = 3 * float(mean_deviation) / asset_std

    trend_segments = (trending != trending.shift()).cumsum() * trending
    starts = close.groupby(trend_segments).transform("first")
    ends = close.groupby(trend_segments).transform("last")
    price_changes = np.abs(ends - starts)
    trend_score = float(price_changes.groupby(trend_segments).mean().mean()) / asset_std

    normalized_neutral = 1 / (1 + neutral_score)
    normalized_trend = 1 / (1 + trend_score)
    score = (normalized_neutral + normalized_trend) / 2

    if indicators:
        score *= np.exp(regime_coherence(mri, indicators) - 1)

    return float(score)


class MarketRegimeIndicator:
    """Combine five statistical/technical regime votes into one score."""

    def __init__(self, params: dict[str, float | int]):
        self.params = params

    def to_json(self) -> str:
        return json.dumps(self.params, indent=2)

    @classmethod
    def from_json(cls, value: str) -> "MarketRegimeIndicator":
        return cls(json.loads(value))

    def calculate(self, data: pd.DataFrame):
        p = self.params
        jobs = {
            "adf": (rolling_adf, data.close, p["adf_length"], p["adf_threshold"]),
            "kpss": (rolling_kpss, data.close, p["kpss_length"], p["kpss_threshold"]),
            "hurst": (rolling_hurst, data.close, p["hurst_length"], p["hurst_threshold"]),
            "adx": (rolling_adx, data, p["adx_length"], p["adx_threshold"]),
            "pp": (rolling_pp, data.close, p["pp_length"], p["pp_threshold"]),
        }

        with ThreadPoolExecutor() as executor:
            futures = {name: executor.submit(func, *args) for name, (func, *args) in jobs.items()}
            indicators = {name: future.result() for name, future in futures.items()}

        mri = sum(indicators.values()) / len(indicators)
        score = evaluate_regime_score(data.close, mri, indicators)
        return mri, indicators, score


def optimize_regime_parameters(
    data_by_asset: list[pd.DataFrame], n_trials: int = 100
) -> tuple[float, dict[str, float | int]]:
    """Search one parameter set across several assets using Optuna + Dask."""
    client = Client()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "adf_length": trial.suggest_int("adf_length", 90, 110),
            "adf_threshold": trial.suggest_float("adf_threshold", -2.0, -1.0),
            "kpss_length": trial.suggest_int("kpss_length", 90, 110),
            "kpss_threshold": trial.suggest_float("kpss_threshold", 0.5, 1.5),
            "hurst_length": trial.suggest_int("hurst_length", 60, 80),
            "hurst_threshold": trial.suggest_float("hurst_threshold", 0.48, 0.52),
            "adx_length": trial.suggest_int("adx_length", 15, 25),
            "adx_threshold": trial.suggest_float("adx_threshold", 20.0, 25.0),
            "pp_length": trial.suggest_int("pp_length", 90, 110),
            "pp_threshold": trial.suggest_float("pp_threshold", -2.5, -1.0),
        }

        futures = [client.submit(MarketRegimeIndicator(params).calculate, data) for data in data_by_asset]
        scores = [result[2] for result in client.gather(futures)]
        return float(np.mean(scores))

    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return float(study.best_value), study.best_params
    finally:
        client.close()
