from .analytics import (
    # Alignment
    align_monthly,
    # Signal evaluation
    bin_by_quantile,
    spread_stat,
    information_coefficient,
    hit_rate,
    eval_signals,
    # Time series
    forward_returns,
    rolling_eval,
    tranche_eval,
    multi_signal_tranche,
    # Visualization
    plot_heatmap,
    plot_bar_by_group,
    plot_time_series_with_fill,
)

from .constituents import fetch as fetch_constituents, save as save_constituents, FUNDS
from .identifiers import add_yahoo_ticker, enrich_with_figi
from .benchmarks import fetch_french, fetch_aqr, fetch_damodaran, FRENCH_DATASETS, AQR_DATASETS, DAMODARAN_DATASETS
from .prices import fetch_prices
from .fundamentals import fetch_fundamentals
from .refinitiv import RefinitivClient
from .climate import fetch_ngfs, load_ngfs, carbon_cost, gdp_haircut, scenario_summary, plot_scenarios, SCENARIOS

__version__ = "0.10.0"
