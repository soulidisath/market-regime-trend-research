# Examples

This directory contains small reproducible examples built from the cleaned research modules.

## BTC-USD regime example

`btc_usd_demo.py` downloads daily BTC-USD OHLC data with `yfinance`, applies a fixed Market Regime Indicator parameter set, and generates two diagnostic figures:

- `output/btc_regimes.png` — BTC-USD price with periods classified as trending (`MRI > 0.5`) shaded in the background.
- `output/btc_regime_components.png` — the five component regime votes (ADF, KPSS, Hurst, ADX, and Phillips-Perron).

Run it from the repository root:

```bash
python examples/btc_usd_demo.py
```

The example uses fixed representative parameters from the ranges explored in the original research. It does **not** re-optimize them on BTC-USD before plotting. The figures are descriptive diagnostics of the regime-classification logic, not evidence of trading profitability.

A GitHub Actions workflow also reruns this example after relevant source changes and refreshes the generated figures when the output changes.

The original work was developed across exploratory notebooks. Rather than publishing notebooks with stale local paths and hidden state, the core methods were moved into `src/` and the examples here are intended to be runnable from a clean environment.
