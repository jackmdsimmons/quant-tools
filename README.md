# quant-tools

Reusable signal evaluation and visualization toolkit for quantitative analysis.

## Install

```bash
pip install git+https://github.com/jackmdsimmons/quant-tools.git@v0.1.0
```

## What's in it

| Function | Description |
|---|---|
| `align_monthly(signal, outcome, lag=1)` | Resample to month-end, lag signal, inner join |
| `bin_by_quantile(signal, outcome, n_bins=5)` | Mean/std/count of outcome per quantile bin |
| `spread_stat(signal, outcome)` | Top-group minus bottom-group mean outcome |
| `information_coefficient(signal, outcome)` | Spearman rank correlation |
| `hit_rate(signal, outcome)` | Fraction where sign(signal) == sign(outcome) |
| `eval_signals(signals_df, outcome, stat_fn)` | Apply any stat to each column in a DataFrame |
| `forward_returns(prices, horizons)` | Multi-horizon forward returns from a price series |
| `rolling_eval(signal, outcome, stat_fn, window)` | Rolling window stat over time |
| `tranche_eval(signal, outcome, stat_fn, tranche_years)` | Stat per N-year period |
| `multi_signal_tranche(signals_df, outcome, stat_fn)` | Tranche eval across multiple signals |
| `plot_heatmap(df, title, ...)` | Annotated heatmap |
| `plot_bar_by_group(values, title, ...)` | Bar chart by group with optional spread annotation |
| `plot_time_series_with_fill(series, title, ...)` | Line chart with green/red fill around zero |

## Usage

```python
from quant_tools.analytics import (
    align_monthly, forward_returns,
    spread_stat, information_coefficient,
    rolling_eval, multi_signal_tranche,
)

# Align signal to outcome (1-month lag)
aligned = align_monthly(my_signal, outcome, lag=1)

# Scalar stats
spread = spread_stat(aligned["signal"], aligned["outcome"])
ic     = information_coefficient(aligned["signal"], aligned["outcome"])

# Rolling IC over time
rolling_ic = rolling_eval(aligned["signal"], aligned["outcome"],
                          information_coefficient, window=36)

# 5-year tranche comparison across multiple signals
table = multi_signal_tranche(signals_df, outcome, spread_stat, tranche_years=5)
```

## Versioning

Pin to a specific version in your `requirements.txt`:

```
quant-tools @ git+https://github.com/jackmdsimmons/quant-tools.git@v0.1.0
```

To upgrade a project, change the tag and run `pip install -r requirements.txt`.
