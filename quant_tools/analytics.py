"""
General-purpose signal evaluation and visualization toolkit.

All analytical functions accept plain pd.Series / pd.DataFrame inputs and are
decoupled from any specific asset class or metric framing. They compose via
`stat_fn` callables, so rolling_eval and tranche_eval work with any scalar
summary (spread_stat, information_coefficient, hit_rate, or custom).

Sections
--------
1. Alignment utilities
2. Signal evaluation  (bin_by_quantile, spread_stat, information_coefficient, hit_rate)
3. Time series eval   (forward_returns, rolling_eval, tranche_eval)
4. Visualization      (plot_heatmap, plot_bar_by_group, plot_time_series_with_fill)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── 1. Alignment utilities ────────────────────────────────────────────────────

def align_monthly(
    signal: pd.Series,
    outcome: pd.Series,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Resample both series to month-end, lag the signal by `lag` periods
    (so signal observed at month T predicts outcome at month T+lag),
    then inner-join and drop NaNs.

    Parameters
    ----------
    signal  : daily or monthly series with DatetimeIndex
    outcome : daily or monthly series with DatetimeIndex
    lag     : months to shift signal forward (default 1 = signal known before outcome)

    Returns
    -------
    DataFrame with columns ['signal', 'outcome'], index = month-end dates
    """
    sig = signal.resample("ME").last().shift(lag)
    out = outcome.resample("ME").last() if outcome.index.freq != "ME" else outcome
    combined = pd.concat([sig.rename("signal"), out.rename("outcome")], axis=1).dropna()
    return combined


# ── 2. Signal evaluation ──────────────────────────────────────────────────────

def bin_by_quantile(
    signal: pd.Series,
    outcome: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Bin signal into n equal-frequency bins, compute mean/std/count of outcome
    per bin. Returns None if signal has fewer unique values than n_bins.

    Parameters
    ----------
    signal  : signal values (already aligned to outcome)
    outcome : outcome values (already aligned to signal)
    n_bins  : number of bins (default 5 = quintiles)

    Returns
    -------
    DataFrame: index = bin labels (Q1..Qn), columns = [mean, std, count]
    """
    if signal.nunique() < n_bins:
        return None

    labels = [f"Q{i+1}" for i in range(n_bins)]
    bins = pd.qcut(signal, n_bins, labels=labels, duplicates="drop")
    grouped = outcome.groupby(bins, observed=True)
    return pd.DataFrame({
        "mean":  grouped.mean(),
        "std":   grouped.std(),
        "count": grouped.count(),
    })


def spread_stat(
    signal: pd.Series,
    outcome: pd.Series,
    lo_pct: float = 0.20,
    hi_pct: float = 0.80,
) -> float:
    """
    Top-group minus bottom-group mean outcome, where groups are defined by
    percentile cutoffs within the provided sample.

    Parameters
    ----------
    signal  : signal values (already aligned to outcome)
    outcome : outcome values (already aligned to signal)
    lo_pct  : lower cutoff (default 0.20 = bottom quintile)
    hi_pct  : upper cutoff (default 0.80 = top quintile)

    Returns
    -------
    float : mean(outcome | signal >= hi_pct) - mean(outcome | signal <= lo_pct)
            Returns NaN if either group is empty or signal has no variance.
    """
    if signal.nunique() < 3:
        return np.nan
    lo = signal.quantile(lo_pct)
    hi = signal.quantile(hi_pct)
    bottom = outcome[signal <= lo].mean()
    top    = outcome[signal >= hi].mean()
    return top - bottom


def information_coefficient(
    signal: pd.Series,
    outcome: pd.Series,
) -> float:
    """
    Spearman rank correlation between signal and outcome.
    Standard IC measure used in factor research; robust to outliers.

    Returns
    -------
    float : correlation in [-1, 1], or NaN if inputs are constant / too short
    """
    aligned = pd.concat([signal, outcome], axis=1).dropna()
    if len(aligned) < 4:
        return np.nan
    ic, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return ic


def hit_rate(
    signal: pd.Series,
    outcome: pd.Series,
) -> float:
    """
    Fraction of observations where sign(signal) == sign(outcome).
    Useful for directional signals (e.g. tsmom).

    Returns
    -------
    float : hit rate in [0, 1]
    """
    aligned = pd.concat([signal, outcome], axis=1).dropna()
    if len(aligned) == 0:
        return np.nan
    correct = (np.sign(aligned.iloc[:, 0]) == np.sign(aligned.iloc[:, 1])).sum()
    return correct / len(aligned)


def eval_signals(
    signals: pd.DataFrame,
    outcome: pd.Series,
    stat_fn=None,
) -> pd.Series:
    """
    Apply a scalar stat function to each column of signals vs. a single outcome.
    Defaults to information_coefficient if stat_fn is None.

    Parameters
    ----------
    signals : DataFrame of signal columns (already aligned to outcome)
    outcome : outcome series (already aligned to signals)
    stat_fn : callable(signal, outcome) -> float

    Returns
    -------
    Series : stat value per signal column
    """
    if stat_fn is None:
        stat_fn = information_coefficient
    return pd.Series(
        {col: stat_fn(signals[col], outcome) for col in signals.columns}
    )


# ── 3. Time series evaluation ─────────────────────────────────────────────────

def forward_returns(
    prices: pd.Series,
    horizons: list[int] = [1, 3, 6, 12],
) -> pd.DataFrame:
    """
    Compute forward returns at multiple month horizons from a price series.
    horizon=1 means next month's return, horizon=3 means next 3 months, etc.

    Returns
    -------
    DataFrame: index = month-end dates, columns = ['fwd_1m', 'fwd_3m', ...]
    """
    monthly = prices.resample("ME").last()
    fwd = pd.DataFrame(index=monthly.index)
    for h in horizons:
        fwd[f"fwd_{h}m"] = monthly.pct_change(h).shift(-h)
    return fwd.dropna()


def rolling_eval(
    signal: pd.Series,
    outcome: pd.Series,
    stat_fn=None,
    window: int = 36,
    min_unique: int = 4,
) -> pd.Series:
    """
    Compute a scalar stat over a rolling window of monthly observations.

    Parameters
    ----------
    signal   : monthly signal series (DatetimeIndex, already lagged if needed)
    outcome  : monthly outcome series (DatetimeIndex)
    stat_fn  : callable(signal_window, outcome_window) -> float
               Defaults to spread_stat.
    window   : number of months per window (default 36)
    min_unique: minimum unique signal values required per window

    Returns
    -------
    Series : stat value at each month-end date (indexed to last date in window)
    """
    if stat_fn is None:
        stat_fn = spread_stat

    combined = pd.concat(
        [signal.rename("signal"), outcome.rename("outcome")], axis=1
    ).dropna()

    results = {}
    for end_idx in range(window, len(combined) + 1):
        w = combined.iloc[end_idx - window : end_idx]
        date = w.index[-1]
        if w["signal"].nunique() < min_unique:
            continue
        results[date] = stat_fn(w["signal"], w["outcome"])

    return pd.Series(results)


def tranche_eval(
    signal: pd.Series,
    outcome: pd.Series,
    stat_fn=None,
    tranche_years: int = 5,
    min_obs: int = 10,
    min_unique: int = 4,
    _tranche_starts=None,
    _tranche_labels=None,
) -> pd.Series:
    """
    Compute a scalar stat within each N-year tranche of monthly observations.

    Parameters
    ----------
    signal        : monthly signal series (DatetimeIndex, already lagged if needed)
    outcome       : monthly outcome series (DatetimeIndex)
    stat_fn       : callable(signal_window, outcome_window) -> float
                    Defaults to spread_stat.
    tranche_years : years per tranche (default 5)
    min_obs       : minimum observations required in a tranche
    min_unique    : minimum unique signal values required in a tranche

    Returns
    -------
    Series : stat value per tranche, indexed by label '1993-1997', etc.
    """
    if stat_fn is None:
        stat_fn = spread_stat

    combined = pd.concat(
        [signal.rename("signal"), outcome.rename("outcome")], axis=1
    ).dropna()

    if _tranche_starts is None:
        start_year = combined.index.year.min()
        end_year   = combined.index.year.max()
        _tranche_starts = list(range(start_year, end_year + 1, tranche_years))
        _tranche_labels = [
            f"{y}-{min(y + tranche_years - 1, end_year)}"
            for y in _tranche_starts
        ]

    def get_label(year):
        for i, y in enumerate(_tranche_starts):
            if year < y + tranche_years:
                return _tranche_labels[i]
        return _tranche_labels[-1]

    combined["tranche"] = combined.index.year.map(get_label)
    results = {}
    for label, grp in combined.groupby("tranche"):
        if len(grp) < min_obs or grp["signal"].nunique() < min_unique:
            continue
        results[label] = stat_fn(grp["signal"], grp["outcome"])

    # Return in chronological order
    return pd.Series({l: results[l] for l in _tranche_labels if l in results})


def multi_signal_tranche(
    signals: pd.DataFrame,
    outcome: pd.Series,
    stat_fn=None,
    tranche_years: int = 5,
) -> pd.DataFrame:
    """
    Apply tranche_eval to each column in signals vs. a single outcome,
    using shared tranche boundaries derived from the outcome date range
    so all signals are sliced into the same periods.

    Returns
    -------
    DataFrame : rows = signals, columns = tranche labels
    """
    # Compute shared tranche boundaries from the common outcome range
    start_year = outcome.index.year.min()
    end_year   = outcome.index.year.max()
    tranche_starts = list(range(start_year, end_year + 1, tranche_years))
    tranche_labels = [
        f"{y}-{min(y + tranche_years - 1, end_year)}"
        for y in tranche_starts
    ]

    rows = {}
    for col in signals.columns:
        rows[col] = tranche_eval(
            signals[col], outcome, stat_fn, tranche_years,
            _tranche_starts=tranche_starts, _tranche_labels=tranche_labels,
        )
    return pd.DataFrame(rows).T


# ── 4. Visualization ──────────────────────────────────────────────────────────

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = ".2f",
    center: float = 0,
    cmap: str = "RdYlGn",
    cbar_label: str = "",
    fname: str = None,
):
    """
    Annotated heatmap with consistent styling.

    Parameters
    ----------
    df         : DataFrame to plot (rows = y-axis, columns = x-axis)
    title      : chart title
    fname      : if provided, save to this path; otherwise display
    """
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.2), max(4, len(df) * 0.7)))
    sns.heatmap(
        df.astype(float),
        annot=True, fmt=fmt, center=center,
        cmap=cmap, ax=ax,
        linewidths=0.5,
        cbar_kws={"label": cbar_label} if cbar_label else {},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {fname}")
    else:
        plt.show()


def plot_bar_by_group(
    values: pd.Series,
    title: str,
    ylabel: str = "",
    colors: list = None,
    annotate_spread: bool = True,
    fname: str = None,
):
    """
    Bar chart of values by group (e.g. mean return per quintile).

    Parameters
    ----------
    values           : Series with group labels as index
    annotate_spread  : if True, annotate Q5-Q1 spread when Q1/Q5 labels exist
    fname            : if provided, save to this path; otherwise display
    """
    default_colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850"]
    bar_colors = colors if colors else default_colors[:len(values)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(values.index, values.values, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    if annotate_spread and "Q5" in values.index and "Q1" in values.index:
        spread = values["Q5"] - values["Q1"]
        ax.text(0.98, 0.02, f"Q5-Q1: {spread:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {fname}")
    else:
        plt.show()


def plot_time_series_with_fill(
    series: pd.Series,
    title: str,
    ylabel: str = "",
    zero_line: bool = True,
    fname: str = None,
):
    """
    Line chart with green fill above zero and red fill below.
    Useful for any spread or IC time series.

    Parameters
    ----------
    series    : time-indexed Series
    fname     : if provided, save to this path; otherwise display
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, color="steelblue", linewidth=1.2)
    if zero_line:
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.fill_between(s.index, s.values, 0, where=s.values > 0, alpha=0.15, color="green")
    ax.fill_between(s.index, s.values, 0, where=s.values < 0, alpha=0.15, color="red")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {fname}")
    else:
        plt.show()
