"""Trend-signal ensemble and robustness utilities.

This module distills the useful parts of the original exploratory TPI research:
combining modular trend signals, tuning indicator parameters across several
assets, searching ensemble weights, and testing sensitivity with OHLC path
permutations.

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


def monte_carlo_ohlc(
    ohlc: pd.DataFrame,
    n_simulations: int = 100,
    seed: int | None = None,
) -> list[pd.DataFrame]:
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
    ordered_columns = ["open", "high", "low", "close"]
    log_data = np.log(ohlc[ordered_columns].astype(float))

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

    The constants reflect the original exploratory scoring logic. This score is
    useful for ranking candidate configurations inside the project; it is not a
    theoretically optimal objective or evidence of expected live performance.
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
    """Wrapper around one directional indicator and its parameter dictionary."""

    name: str
    function: Callable
    params: dict

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if self.name == "trendilo":
            hl2 = data["hl2"] if "hl2" in data else (data["high"] + data["low"]) / 2
            return self.function(data.close, hl2, **self.params)
        if self.name in {"dema_dmi", "vii_stop", "lsma_atr"}:
            return self.function(data.close, data.high, data.low, **self.params)
        return self.function(data.close, **self.params)


MODEL_FUNCTIONS: dict[str, Callable] = {
    "trendilo": indicators.trendilo,
    "dema_dmi": indicators.dema_dmi,
    "kama": indicators.kama,
    "stc": indicators.stc,
    "vii_stop": indicators.vii_stop,
    "rti": indicators.rti,
    "lrsi": indicators.lrsi,
    "lsma_atr": indicators.lsma_atr,
    "hullloop": indicators.hullloop,
    "aamdmom": indicators.aamdmom,
}

DEFAULT_MODELS = [SignalModel(name, function, {}) for name, function in MODEL_FUNCTIONS.items()]


def _suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
    """Suggest parameters using the search ranges from the legacy experiments."""
    prefix = lambda name: f"{model_name}__{name}"

    if model_name == "trendilo":
        return {
            "lsma_length": trial.suggest_int(prefix("lsma_length"), 10, 70),
            "offset": trial.suggest_int(prefix("offset"), -10, 10),
            "smooth": trial.suggest_int(prefix("smooth"), 5, 15),
            "hma_length": trial.suggest_int(prefix("hma_length"), 50, 150),
        }
    if model_name == "dema_dmi":
        return {
            "len_dema": trial.suggest_int(prefix("len_dema"), 10, 90),
            "adx_smoothing_len": trial.suggest_int(prefix("adx_smoothing_len"), 2, 12),
            "di_len": trial.suggest_int(prefix("di_len"), 2, 102),
        }
    if model_name == "kama":
        return {
            "fast_period": trial.suggest_int(prefix("fast_period"), 2, 20),
            "slow_period": trial.suggest_int(prefix("slow_period"), 10, 100),
            "er_period": trial.suggest_int(prefix("er_period"), 2, 10),
            "norm_period": trial.suggest_int(prefix("norm_period"), 10, 100),
        }
    if model_name == "stc":
        return {
            "stc_length": trial.suggest_int(prefix("stc_length"), 10, 80),
            "fast_length": trial.suggest_int(prefix("fast_length"), 10, 40),
            "slow_length": trial.suggest_int(prefix("slow_length"), 40, 100),
            "factor": trial.suggest_float(prefix("factor"), 0.01, 0.5),
        }
    if model_name == "vii_stop":
        return {
            "length": trial.suggest_int(prefix("length"), 10, 100),
            "multiplier": trial.suggest_float(prefix("multiplier"), 2.0, 4.0),
        }
    if model_name == "rti":
        return {
            "trend_data_count": trial.suggest_int(prefix("trend_data_count"), 50, 150),
            "trend_sensitivity_percentage": trial.suggest_int(prefix("trend_sensitivity_percentage"), 80, 98),
            "signal_length": trial.suggest_int(prefix("signal_length"), 10, 30),
            "long_threshold": trial.suggest_int(prefix("long_threshold"), 50, 80),
            "short_threshold": trial.suggest_int(prefix("short_threshold"), 30, 50),
        }
    if model_name == "lrsi":
        return {
            "alpha": trial.suggest_float(prefix("alpha"), 0.1, 0.3),
            "long_threshold": trial.suggest_float(prefix("long_threshold"), 0.5, 0.8),
            "short_threshold": trial.suggest_float(prefix("short_threshold"), 0.3, 0.5),
        }
    if model_name == "lsma_atr":
        return {
            "len_lsma": trial.suggest_int(prefix("len_lsma"), 50, 100),
            "len_atr": trial.suggest_int(prefix("len_atr"), 10, 50),
            "atr_multiplier": trial.suggest_float(prefix("atr_multiplier"), 0.5, 2.5),
        }
    if model_name == "hullloop":
        return {
            "length": trial.suggest_int(prefix("length"), 4, 15),
            "a": trial.suggest_int(prefix("a"), 1, 10),
            "b": trial.suggest_int(prefix("b"), 45, 55),
        }
    if model_name == "aamdmom":
        return {
            "len_momentum": trial.suggest_int(prefix("len_momentum"), 4, 15),
            "len_percentile": trial.suggest_int(prefix("len_percentile"), 40, 60),
        }
    raise KeyError(f"No parameter search space defined for {model_name!r}.")


def _params_from_flat_dict(flat_params: dict[str, float | int], model_name: str) -> dict:
    prefix = f"{model_name}__"
    return {key.removeprefix(prefix): value for key, value in flat_params.items() if key.startswith(prefix)}


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
        if len(use_weights) != len(self.models) or use_weights.sum() <= 0:
            raise ValueError("Weights must match the number of models and sum to a positive value.")
        use_weights = use_weights / use_weights.sum()
        weighted = components.to_numpy() @ use_weights
        return pd.Series(np.sign(weighted), index=data.index).replace(0, np.nan).ffill()

    def backtest(self, data: pd.DataFrame, allow_shorts: bool = True):
        return backtest_signals(data, self.signal(data), allow_shorts=allow_shorts)

    def optimize_parameters(
        self,
        datasets: Sequence[pd.DataFrame],
        n_trials: int = 100,
    ) -> tuple[dict[str, dict], float]:
        """Tune the selected models' parameters against the mean cross-asset score.

        This restores the first optimization layer from the legacy TPI research.
        The model subset is now chosen explicitly when constructing the ensemble
        instead of being sampled randomly inside the optimizer.
        """
        if not datasets:
            raise ValueError("At least one dataset is required.")

        client = Client()

        def score_dataset(data: pd.DataFrame, candidate_models: list[SignalModel]) -> float:
            candidate = TrendEnsemble(candidate_models)
            signal = candidate.signal(data)
            portfolio = backtest_signals(data, signal, allow_shorts=True)
            return research_score(portfolio)

        def objective(trial: optuna.Trial) -> float:
            candidate_models = [
                SignalModel(model.name, model.function, _suggest_model_params(trial, model.name))
                for model in self.models
            ]
            futures = [client.submit(score_dataset, data, candidate_models) for data in datasets]
            return float(np.mean(client.gather(futures)))

        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best = {
                model.name: _params_from_flat_dict(study.best_params, model.name)
                for model in self.models
            }
            self.models = [SignalModel(model.name, model.function, best[model.name]) for model in self.models]
            return best, float(study.best_value)
        finally:
            client.close()

    def optimize_weights(
        self,
        datasets: Sequence[pd.DataFrame],
        n_trials: int = 100,
    ) -> tuple[np.ndarray, float]:
        """Search one normalized weight vector across several assets."""
        if not datasets:
            raise ValueError("At least one dataset is required.")

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

    def monte_carlo_scores(
        self,
        data: pd.DataFrame,
        n_simulations: int = 100,
        seed: int | None = None,
    ) -> pd.Series:
        """Evaluate the fixed ensemble on permuted OHLC paths."""
        scores = []
        for simulated in monte_carlo_ohlc(data, n_simulations=n_simulations, seed=seed):
            signal = self.signal(simulated)
            portfolio = backtest_signals(simulated, signal, allow_shorts=True)
            scores.append(research_score(portfolio))
        return pd.Series(scores, name="research_score")
