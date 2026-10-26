"""Trend-signal ensemble and robustness utilities.

This module distills the useful parts of the original exploratory TPI research:
combining modular trend signals, evaluating them with a simple vectorbt backtest,
searching ensemble weights across several assets, and testing sensitivity with
OHLC path permutations.

The code is intended for research and portfolio demonstration, not live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import optuna
import pandas as pd
import vectorbt as vbt
from dask.distributed import Client

from . import indicators


def monte_carlo_ohlc(ohlc: pd.DataFrame, n_simulations: int = 100, seed: int | None = None) -> list[pd.DataFrame]:
    """Create synthetic OHLC paths by permuting inter- and intra-bar log returns.

    This preserves the empirical building blocks of the observed bars while
    changing their temporal ordering. It is used here as a sensitivity test,
    not as a substitute for out-of-sample validation.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(ohlc.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if len(ohlc) < 3:
        raise ValueError("At least three rows are required for path permutation.")

    rng = np.random.default_rng(seed)
    log_data = np.log(ohlc[list(required)].astype(float))

    delta_h = (log_data["high"] - log_data["open"]).to_numpy()
    delta_l = (log_data["low"] - log_data["open"]).to_numpy()
    delta_c = (log_data["close"] - log_data["open"]).to_numpy()
    inter = log_data["open"].iloc[1:].to_numpy() - log_data["close"].iloc[:-1].to_numpy()

    simulations: list[pd.DataFrame] = []
    for _ in range(n_simulations):
        inter_perm = rng.permutation(inter)
        intra_idx = rng.permutation(np.arange(1, len(ohlc)))

        new_open = [log_data["open"].iloc[0]]
        new_high = [log_data["high"].iloc[0]]
        new_low = [log_data["low"].iloc[0]]
        new_close = [log_data["close"].iloc[0]]

        last_close = new_close[0]
        for step, idx in enumerate(intra_idx):
            new_o = last_close + inter_perm[step]
            new_h = new_o + delta_h[idx]
            new_l = new_o + delta_l[idx]
            new_c = new_o + delta_c[idx]
            new_open.append(new_o)
            new_high.append(new_h)
            new_low.append(new_l)
            new_close.append(new_c)
            last_close = new_c

        simulated = pd.DataFrame(
            {
                "open": np.exp(new_open),
                "high": np.exp(new_high),
                "low": np.exp(new_low),
                "close": np.exp(new_close),
            },
            index=ohlc.index,
        )
        simulated["hl2"] = (simulated["high"] + simulated["low"]) / 2
        simulations.append(simulated)

    return simulations


def backtest_signals(data: pd.DataFrame, signals: pd.Series, allow_shorts: bool = True):
    """Run a deliberately simple signal-flip backtest with vectorbt."""
    signals = signals.reindex(data.index).ffill().fillna(0)
    long_entries = signals.eq(1)
    long_exits = signals.eq(-1)
    short_entries = signals.eq(-1) if allow_shorts else pd.Series(False, index=data.index)
    short_exits = signals.eq(1) if allow_shorts else pd.Series(False, index=data.index)

    return vbt.Portfolio.from_signals(
        data["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )


def portfolio_metrics(portfolio) -> dict[str, float]:
    """Extract the metrics used in the original comparison heuristic."""
    stats = portfolio.stats()
    return {
        "max_drawdown": float(portfolio.max_drawdown()),
        "sharpe": float(portfolio.sharpe_ratio()),
        "sortino": float(portfolio.sortino_ratio()),
        "omega": float(portfolio.omega_ratio()),
        "profit_factor": float(stats.get("Profit Factor", np.nan)),
        "win_rate": float(stats.get("Win Rate [%]", np.nan)) / 100.0,
    }


def research_score(portfolio) -> float:
    """Heuristic score used to compare configurations inside the experiments.

    It intentionally combines several risk/return diagnostics rather than
    presenting any one metric as proof of a superior strategy.
    """
    metrics = portfolio_metrics(portfolio)
    values = np.array(
        [
            -metrics["max_drawdown"],
            metrics["sharpe"] / 2.0,
            metrics["sortino"] / 2.9,
            metrics["omega"] / 1.31,
            min(metrics["profit_factor"], 10.0) / 4.0,
            metrics["win_rate"] / 0.65,
        ],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("-inf")


@dataclass
class SignalModel:
    """Small wrapper around an indicator function and its parameter dictionary."""

    name: str
    function: Callable
    params: dict

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if self.name == "trendilo":
            return self.function(data.close, data.hl2, **self.params)
        if self.name in {"dema_dmi", "vii_stop", "lsma_atr"}:
            return self.function(data.close, data.high, data.low, **self.params)
        return self.function(data.close, **self.params)


DEFAULT_MODELS = [
    SignalModel("trendilo", indicators.trendilo, {}),
    SignalModel("dema_dmi", indicators.dema_dmi, {}),
    SignalModel("kama", indicators.kama, {}),
    SignalModel("stc", indicators.stc, {}),
    SignalModel("vii_stop", indicators.vii_stop, {}),
    SignalModel("rti", indicators.rti, {}),
    SignalModel("lrsi", indicators.lrsi, {}),
    SignalModel("lsma_atr", indicators.lsma_atr, {}),
    SignalModel("hullloop", indicators.hullloop, {}),
    SignalModel("aamdmom", indicators.aamdmom, {}),
]


class TrendEnsemble:
    """Weighted combination of modular directional signals."""

    def __init__(self, models: Sequence[SignalModel], weights: Sequence[float] | None = None):
        if not models:
            raise ValueError("At least one signal model is required.")
        self.models = list(models)
        if weights is None:
            self.weights = np.ones(len(models), dtype=float) / len(models)
        else:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(models) or weights.sum() <= 0:
                raise ValueError("Weights must match the number of models and sum to a positive value.")
            self.weights = weights / weights.sum()

    def component_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        values = {model.name: model.calculate(data) for model in self.models}
        return pd.DataFrame(values, index=data.index).ffill()

    def signal(self, data: pd.DataFrame, weights: Sequence[float] | None = None) -> pd.Series:
        components = self.component_signals(data)
        use_weights = self.weights if weights is None else np.asarray(weights, dtype=float)
        use_weights = use_weights / use_weights.sum()
        weighted = components.to_numpy() @ use_weights
        return pd.Series(np.sign(weighted), index=data.index).replace(0, np.nan).ffill()

    def backtest(self, data: pd.DataFrame, allow_shorts: bool = True):
        return backtest_signals(data, self.signal(data), allow_shorts=allow_shorts)

    def optimize_weights(self, datasets: Sequence[pd.DataFrame], n_trials: int = 100) -> tuple[np.ndarray, float]:
        """Search one normalized weight vector across several assets."""
        client = Client()

        def score_dataset(data: pd.DataFrame, weights: np.ndarray) -> float:
            signal = self.signal(data, weights)
            return research_score(backtest_signals(data, signal, allow_shorts=True))

        def objective(trial: optuna.Trial) -> float:
            raw = np.array(
                [trial.suggest_float(f"weight_{i}", 0.0, 1.0) for i in range(len(self.models))],
                dtype=float,
            )
            if raw.sum() == 0:
                return -1e9
            weights = raw / raw.sum()
            futures = [client.submit(score_dataset, data, weights) for data in datasets]
            return float(np.mean(client.gather(futures)))

        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            raw = np.array([study.best_params[f"weight_{i}"] for i in range(len(self.models))])
            self.weights = raw / raw.sum()
            return self.weights.copy(), float(study.best_value)
        finally:
            client.close()

    def monte_carlo_scores(self, data: pd.DataFrame, n_simulations: int = 100, seed: int | None = None) -> pd.Series:
        """Evaluate the fixed ensemble on permuted OHLC paths."""
        scores = []
        for simulated in monte_carlo_ohlc(data, n_simulations=n_simulations, seed=seed):
            signal = self.signal(simulated)
            portfolio = backtest_signals(simulated, signal, allow_shorts=True)
            scores.append(research_score(portfolio))
        return pd.Series(scores, name="research_score")
