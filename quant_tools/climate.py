"""
NGFS Climate Scenario data via the IIASA pyam API.

Source: NGFS Phase 5 (November 2024) — the current standard for climate
scenario analysis in financial risk assessment.

No credentials required — the NGFS database is publicly accessible.

Installation
------------
    pip install pyam-iamc

Usage
-----
    from quant_tools.climate import fetch_ngfs, carbon_cost, gdp_haircut

    # Download all key scenario variables
    ngfs = fetch_ngfs()                          # returns dict of DataFrames

    # Carbon price paths (USD/tCO2, 2020–2050)
    carbon = ngfs["carbon_price"]

    # Apply carbon cost to a portfolio's emissions intensity
    cost_df = carbon_cost(
        carbon_prices=carbon,
        emissions=fund[["ticker", "scope1_intensity"]],   # tCO2 / USD revenue
        revenue=fund["revenue"],
    )

    # GDP haircut by region/scenario
    gdp = ngfs["gdp_impact"]

NGFS Phase 5 scenarios
----------------------
    Orderly     : "Net Zero 2050", "Below 2°C", "Low Demand"
    Disorderly  : "Divergent Net Zero", "Delayed Transition"
    Hot house   : "Fragmented World", "Current Policies", "NDCs"

Key variables pulled
--------------------
    carbon_price  : Price|Carbon             (USD/tCO2)
    emissions     : Emissions|CO2            (Gt CO2/yr, World)
    temperature   : Temperature|Global Mean  (°C above pre-industrial)
    gdp_impact    : GDP|PPP                  (billion USD PPP, by region)
    energy_mix    : Primary Energy|*         (EJ/yr)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._defaults import _DEFAULT_DATA_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

NGFS_INSTANCE = "ngfs_phase_5"

# Canonical scenario labels and their risk category
SCENARIOS = {
    # Orderly
    "Net Zero 2050":     "orderly",
    "Below 2°C":         "orderly",
    "Low Demand":        "orderly",
    # Disorderly
    "Divergent Net Zero":"disorderly",
    "Delayed Transition":"disorderly",
    # Hot house world
    "Fragmented World":  "hot_house",
    "Current Policies":  "hot_house",
    "NDCs":              "hot_house",
}

SCENARIO_COLORS = {
    "Net Zero 2050":      "#10b981",
    "Below 2°C":          "#34d399",
    "Low Demand":         "#6ee7b7",
    "Divergent Net Zero": "#f59e0b",
    "Delayed Transition": "#fbbf24",
    "Fragmented World":   "#ef4444",
    "Current Policies":   "#dc2626",
    "NDCs":               "#f87171",
}

# Variables to download (pyam name → our column name)
_VARIABLES = {
    "Price|Carbon":            "carbon_price",
    "Emissions|CO2":           "co2_emissions",
    "Temperature|Global Mean": "temperature",
    "GDP|PPP":                 "gdp",
    "Primary Energy|Coal":     "energy_coal",
    "Primary Energy|Gas":      "energy_gas",
    "Primary Energy|Oil":      "energy_oil",
    "Primary Energy|Wind":     "energy_wind",
    "Primary Energy|Solar":    "energy_solar",
}

# NGFS regions → ISO-style labels used in our constituents data
REGION_MAP = {
    "World":          "World",
    "R5ASIA":         "Asia",
    "R5LAM":          "Latin America",
    "R5MAF":          "Middle East & Africa",
    "R5OECD90+EU":    "OECD+EU",
    "R5REF":          "Reforming Economies",
}


# ── Core download ─────────────────────────────────────────────────────────────

def fetch_ngfs(
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
    regions: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2100,
    output_dir: str | None = _DEFAULT_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Download NGFS Phase 5 scenario data from the IIASA database.

    Parameters
    ----------
    scenarios  : list of scenario names to fetch (default: all 8)
    variables  : list of pyam variable names (default: core financial set)
    regions    : list of NGFS regions (default: ['World'])
    start_year : first year to include (default: 2020)
    end_year   : last year to include (default: 2100)
    output_dir : save CSVs here (default: quant-tools/data/). Pass None to skip.

    Returns
    -------
    Dict with keys matching _VARIABLES values:
        'carbon_price', 'co2_emissions', 'temperature', 'gdp',
        'energy_coal', 'energy_gas', 'energy_oil', 'energy_wind', 'energy_solar'
    Each value is a DataFrame with columns = scenario names, index = year.
    For regional variables (gdp, energy) index is MultiIndex (region, year).
    """
    try:
        import pyam
    except ImportError:
        raise ImportError("Run: pip install pyam-iamc")

    scenarios = scenarios or list(SCENARIOS.keys())
    variables = variables or list(_VARIABLES.keys())
    regions   = regions   or ["World"]

    print(f"Connecting to NGFS Phase 5 (IIASA)...")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Variables : {len(variables)}")
    print(f"  Regions   : {regions}")

    raw = pyam.read_iiasa(
        NGFS_INSTANCE,
        scenario=scenarios,
        variable=variables,
        region=regions,
    )

    print(f"  Downloaded {raw.shape[0]:,} data points across "
          f"{len(raw.scenario.unique())} scenarios")

    # Build tidy per-variable DataFrames
    results: dict[str, pd.DataFrame] = {}
    years = list(range(start_year, end_year + 1, 5))

    for pyam_var, col_name in _VARIABLES.items():
        if pyam_var not in variables:
            continue

        try:
            sub = raw.filter(variable=pyam_var)
            if sub.empty:
                continue

            # timeseries: models × years — average across models per scenario
            ts = sub.timeseries()
            ts = ts.reset_index()

            # Keep years in range
            year_cols = [c for c in ts.columns if isinstance(c, int)
                         and start_year <= c <= end_year]

            # Average across models, pivot to scenario columns
            df = (
                ts[["scenario"] + year_cols]
                .groupby("scenario")[year_cols]
                .mean()
                .T
            )
            df.index.name = "year"
            df = df[[s for s in scenarios if s in df.columns]]

            results[col_name] = df

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fname = os.path.join(output_dir, f"ngfs_{col_name}.csv")
                df.to_csv(fname)

            print(f"  ✓ {col_name:20s} {df.shape}")

        except Exception as e:
            print(f"  ✗ {col_name}: {e}")

    if output_dir:
        print(f"\nSaved to {output_dir}/ngfs_*.csv")

    return results


def load_ngfs(data_dir: str = _DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load previously saved NGFS CSVs from disk (avoids re-downloading).

    Returns same dict structure as fetch_ngfs().
    """
    results = {}
    for col_name in _VARIABLES.values():
        fname = os.path.join(data_dir, f"ngfs_{col_name}.csv")
        if os.path.exists(fname):
            df = pd.read_csv(fname, index_col=0)
            df.index = df.index.astype(int)
            results[col_name] = df
    if not results:
        raise FileNotFoundError(
            f"No NGFS files found in {data_dir}. Run fetch_ngfs() first."
        )
    print(f"Loaded {len(results)} NGFS datasets from {data_dir}")
    return results


# ── Financial application layer ───────────────────────────────────────────────

def carbon_cost(
    carbon_prices: pd.DataFrame,
    emissions_intensity: pd.Series,
    revenue: pd.Series,
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute implied carbon cost burden per company per scenario.

    Parameters
    ----------
    carbon_prices      : DataFrame from fetch_ngfs()['carbon_price']
                         index=year, columns=scenario (USD/tCO2)
    emissions_intensity: Series indexed by ticker — Scope 1+2 tCO2 per USD revenue
    revenue            : Series indexed by ticker — annual revenue (USD)
    target_years       : years to compute cost for (default: [2030, 2040, 2050])

    Returns
    -------
    DataFrame with MultiIndex (ticker, year), columns = scenarios.
    Values = carbon cost as % of revenue.
    """
    target_years = target_years or [2030, 2040, 2050]
    tickers = emissions_intensity.index.intersection(revenue.index)

    rows = []
    for year in target_years:
        if year not in carbon_prices.index:
            # Interpolate if needed
            yr_prices = carbon_prices.reindex(
                sorted(set(carbon_prices.index) | {year})
            ).interpolate().loc[year]
        else:
            yr_prices = carbon_prices.loc[year]

        for ticker in tickers:
            intensity = emissions_intensity.get(ticker, np.nan)
            rev = revenue.get(ticker, np.nan)
            if pd.isna(intensity) or pd.isna(rev) or rev == 0:
                continue
            # absolute emissions (tCO2)
            abs_emissions = intensity * rev
            # cost per scenario
            cost_usd = abs_emissions * yr_prices  # USD
            cost_pct = cost_usd / rev             # fraction of revenue
            row = {"ticker": ticker, "year": year}
            row.update(cost_pct.to_dict())
            rows.append(row)

    result = pd.DataFrame(rows).set_index(["ticker", "year"])
    return result


def gdp_haircut(
    gdp_paths: pd.DataFrame,
    country_weights: pd.Series,
    baseline_scenario: str = "Current Policies",
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute weighted-average GDP haircut vs baseline for a portfolio's
    geographic exposure.

    Parameters
    ----------
    gdp_paths        : DataFrame from fetch_ngfs()['gdp'] (World-level)
    country_weights  : Series of portfolio weight by country (sums to 1)
    baseline_scenario: scenario to use as baseline (default: 'Current Policies')
    target_years     : years to compute haircut (default: [2030, 2040, 2050])

    Returns
    -------
    DataFrame: index=year, columns=scenario, values=GDP haircut (fraction).
    """
    target_years = target_years or [2030, 2040, 2050]
    scenarios = [s for s in gdp_paths.columns if s != baseline_scenario]
    baseline = gdp_paths[baseline_scenario]

    rows = []
    for year in target_years:
        if year not in gdp_paths.index:
            continue
        row = {"year": year}
        for scen in scenarios:
            haircut = (gdp_paths.loc[year, scen] - baseline.loc[year]) / baseline.loc[year].abs()
            row[scen] = haircut
        rows.append(row)

    return pd.DataFrame(rows).set_index("year")


def scenario_summary(ngfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One-row-per-scenario summary table with key 2030/2050 metrics.

    Returns DataFrame with columns:
        scenario, category, carbon_price_2030, carbon_price_2050,
        temperature_2100, co2_reduction_2050 (vs 2020)
    """
    rows = []
    for scen, category in SCENARIOS.items():
        row = {"scenario": scen, "category": category}

        cp = ngfs.get("carbon_price")
        if cp is not None and scen in cp.columns:
            row["carbon_price_2030"] = cp.loc[2030, scen] if 2030 in cp.index else np.nan
            row["carbon_price_2050"] = cp.loc[2050, scen] if 2050 in cp.index else np.nan

        temp = ngfs.get("temperature")
        if temp is not None and scen in temp.columns:
            row["temperature_2100"] = temp.loc[2100, scen] if 2100 in temp.index else np.nan

        em = ngfs.get("co2_emissions")
        if em is not None and scen in em.columns:
            base = em.loc[2020, scen] if 2020 in em.index else np.nan
            end  = em.loc[2050, scen] if 2050 in em.index else np.nan
            row["co2_reduction_2050"] = (end - base) / abs(base) if pd.notna(base) and base != 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("scenario")
    return df


# ── Convenience ───────────────────────────────────────────────────────────────

def plot_scenarios(
    ngfs: dict[str, pd.DataFrame],
    variable: str = "carbon_price",
    end_year: int = 2060,
    ax=None,
):
    """
    Quick plot of a variable across all scenarios with standard NGFS colours.

    Parameters
    ----------
    ngfs     : dict from fetch_ngfs() or load_ngfs()
    variable : key in ngfs dict (e.g. 'carbon_price', 'temperature')
    end_year : clip x-axis at this year
    ax       : matplotlib axes (creates new figure if None)
    """
    import matplotlib.pyplot as plt

    df = ngfs.get(variable)
    if df is None:
        raise KeyError(f"'{variable}' not in ngfs dict. Available: {list(ngfs)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    df_plot = df[df.index <= end_year]

    for scen in df_plot.columns:
        color    = SCENARIO_COLORS.get(scen, "#888888")
        category = SCENARIOS.get(scen, "")
        ls = "--" if category == "disorderly" else ":"  if category == "hot_house" else "-"
        ax.plot(df_plot.index, df_plot[scen], label=scen,
                color=color, lw=2, linestyle=ls)

    labels = {
        "carbon_price":  "Carbon Price (USD/tCO₂)",
        "co2_emissions": "CO₂ Emissions (Gt CO₂/yr)",
        "temperature":   "Temperature Anomaly (°C)",
        "gdp":           "GDP|PPP (billion USD)",
    }
    ax.set_ylabel(labels.get(variable, variable))
    ax.set_title(f"NGFS Phase 5 — {variable.replace('_', ' ').title()}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    return ax
