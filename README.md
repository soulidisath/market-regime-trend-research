# Market Regime & Trend Research

Personal research project exploring **trend-signal ensembles**, **market-regime classification**, parameter search across multiple assets, and sensitivity to alternate historical price paths.

The repository is a cleaned presentation of exploratory work developed while studying systematic trading. It is **not production trading infrastructure** and it does not claim that the methods produce persistent alpha.

## Research questions

- Can several trend indicators be combined into a more stable composite signal?
- Can unit-root, long-memory, and directional-strength measures help separate persistent/trending periods from more mean-reverting periods?
- How sensitive are optimized configurations to the selected assets and to the ordering of historical price moves?

## What is implemented

### 1. Trend-signal ensemble

`src/trend_ensemble.py` contains the original research framework, cleaned for presentation. It:

- combines multiple trend indicators into composite signals;
- evaluates simple long/short backtests with `vectorbt`;
- compares configurations with drawdown, Sharpe, Sortino, Omega ratio, profit factor, win rate, and a custom signal-coherence term;
- searches indicator parameters and ensemble weights with **Optuna**; and
- parallelizes cross-asset evaluation with **Dask**.

The underlying indicator implementations are in `src/indicators.py`. Several were adapted from public TradingView indicator ideas; links to the original scripts are retained in the relevant docstrings rather than presenting every formula as original work.

### 2. Market Regime Indicator (MRI)

`src/market_regime.py` combines five binary regime measures:

- Augmented Dickey-Fuller (ADF)
- KPSS
- Phillips-Perron
- rolling Hurst exponent
- Average Directional Index (ADX)

Their votes are averaged into a simple regime score. In these experiments, values above `0.5` are treated as a trending regime and values at or below `0.5` as a more mean-reverting regime. A custom heuristic score was used to compare parameter sets across several assets.

### 3. Monte Carlo path permutations

The trend framework includes an OHLC permutation routine that separately shuffles inter-bar and intra-bar log-return components before reconstructing synthetic price paths. I used this as a **sensitivity test**: does an optimized configuration remain reasonable when the temporal ordering of the observed moves is changed?

This should not be confused with genuine out-of-sample validation.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── indicators.py
│   ├── market_regime.py
│   └── trend_ensemble.py
└── examples/
    └── README.md
```

## Running the code

Create a virtual environment and install the research dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The modules expect OHLC data in a pandas DataFrame with lowercase columns such as `open`, `high`, `low`, and `close`. Some trend functions also use `hl2`.

## Important limitations

This work was exploratory. In particular:

- optimization on historical data can overfit;
- the Monte Carlo procedure is a robustness/sensitivity exercise, not out-of-sample evidence;
- the simple backtest helper does not include transaction costs by default;
- regime labels are based on heuristic combinations of statistical and technical measures rather than a formally identified latent-state model; and
- no result in this repository should be interpreted as a claim of live or repeatable trading profitability.

Those limitations are part of the point of the project: the goal was to test ideas, understand failure modes, and make the research process more systematic.
