# Examples

This directory contains small examples built from the cleaned research modules.

## BTC-USD regime example

`btc_usd_demo.py` downloads daily BTC-USD OHLC data with `yfinance`, applies a fixed Market Regime Indicator parameter set, and generates two diagnostic figures:

- `output/btc_regimes.png` — BTC-USD price with periods classified as trending (`MRI > 0.5`) shaded in the background.
- `output/btc_regime_components.png` — the five component regime votes (ADF, KPSS, Hurst, ADX, and Phillips-Perron).

Run it from the repository root:

```bash
python examples/btc_usd_demo.py
```

The example uses fixed representative parameters from the ranges explored in the original research. It does **not** re-optimize them on BTC-USD before plotting. The figures are descriptive diagnostics of the regime-classification logic, not evidence of trading profitability.

A GitHub Actions workflow reruns this example after relevant source changes and refreshes the generated figures when the output changes.

## Trend-ensemble example

`trend_ensemble_demo.py` downloads BTC-USD and visualizes the cleaned multi-indicator ensemble using fixed default parameters. It generates a price/state-change chart and a panel of the individual directional votes.

```bash
python examples/trend_ensemble_demo.py
```

This script deliberately does **not** tune the model on BTC-USD immediately before displaying the same period. The separate research methods `TrendEnsemble.optimize_parameters()` and `TrendEnsemble.optimize_weights()` can be used on explicitly chosen training datasets when running optimization experiments.

The original TPI work was developed across exploratory notebooks and scripts. Rather than publish those files with stale local state, broken helper calls, and hidden assumptions, the core methods have been moved into `src/` and rewritten around explicit inputs.
