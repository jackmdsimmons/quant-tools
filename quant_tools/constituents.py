"""
Download iShares ETF constituent holdings.

Supports any iShares fund by passing a config dict. ACWI is the default.

Usage (as a library)
---------------------
    from quant_tools.constituents import fetch, save, FUNDS
    df = fetch("ACWI")          # 2,275 global equities
    save(df, "ACWI", "data/")

Usage (as a script)
--------------------
    python -m quant_tools.constituents --ticker ACWI
    python -m quant_tools.constituents --all

CSV columns (after cleaning)
-----------------------------
    ticker, name, sector, asset_class, market_value, weight_pct,
    notional_value, quantity, price, country, exchange, currency,
    fx_rate, market_currency, as_of_date
"""

import argparse
import io
import os
import re
from datetime import datetime

import pandas as pd
import requests

# ── Fund registry ─────────────────────────────────────────────────────────────
# Add more iShares funds here as needed.
FUNDS = {
    "ACWI": {
        "name": "iShares MSCI ACWI ETF",
        "url": (
            "https://www.ishares.com/us/products/239600/"
            "ishares-msci-acwi-etf/1467271812596.ajax"
            "?fileType=csv&fileName=ACWI_holdings&dataType=fund"
        ),
    },
    "URTH": {
        "name": "iShares MSCI World ETF",
        "url": (
            "https://www.ishares.com/us/products/239696/"
            "ishares-msci-world-etf/1467271812596.ajax"
            "?fileType=csv&fileName=URTH_holdings&dataType=fund"
        ),
    },
    "EEM": {
        "name": "iShares MSCI Emerging Markets ETF",
        "url": (
            "https://www.ishares.com/us/products/239637/"
            "ishares-msci-emerging-markets-etf/1467271812596.ajax"
            "?fileType=csv&fileName=EEM_holdings&dataType=fund"
        ),
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

COLUMN_RENAME = {
    "Ticker":         "ticker",
    "Name":           "name",
    "Sector":         "sector",
    "Asset Class":    "asset_class",
    "Market Value":   "market_value",
    "Weight (%)":     "weight_pct",
    "Notional Value": "notional_value",
    "Quantity":       "quantity",
    "Price":          "price",
    "Location":       "country",
    "Exchange":       "exchange",
    "Currency":       "currency",
    "FX Rate":        "fx_rate",
    "Market Currency":"market_currency",
    "Accrual Date":   "accrual_date",
}


def _parse_as_of(raw_text: str) -> str:
    """Extract 'as of' date string from the CSV header metadata."""
    match = re.search(r'Fund Holdings as of,"?([^"\n]+)"?', raw_text)
    if match:
        try:
            return datetime.strptime(match.group(1).strip(), "%b %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            return match.group(1).strip()
    return "unknown"


def fetch(ticker: str = "ACWI", equity_only: bool = True) -> pd.DataFrame:
    """
    Download iShares ETF holdings and return a clean DataFrame.

    Parameters
    ----------
    ticker      : fund ticker, must be in FUNDS registry (default 'ACWI')
    equity_only : if True, filter to Asset Class == 'Equity' (default True)

    Returns
    -------
    DataFrame with standardised columns plus 'as_of_date'
    """
    ticker = ticker.upper()
    if ticker not in FUNDS:
        raise ValueError(f"Unknown ticker '{ticker}'. Available: {list(FUNDS)}")

    fund = FUNDS[ticker]
    print(f"Downloading {fund['name']} holdings...")

    response = requests.get(fund["url"], headers=HEADERS, timeout=30)
    response.raise_for_status()
    raw = response.text

    as_of = _parse_as_of(raw)
    print(f"As of: {as_of}")

    # Find header row — contains "Ticker" as first field
    lines = raw.splitlines()
    header_idx = next(
        i for i, line in enumerate(lines)
        if line.startswith("Ticker,") or line.startswith('"Ticker",')
    )

    # Read from header row; pandas will stop at empty lines naturally
    df = pd.read_csv(
        io.StringIO(raw),
        skiprows=header_idx,
        thousands=",",
        on_bad_lines="skip",
    )

    # Drop footer rows — they appear after the last real data row and have
    # NaN in the Ticker column or contain disclaimer text
    if "Ticker" in df.columns:
        df = df[df["Ticker"].notna()]
        df = df[~df["Ticker"].str.startswith("The ", na=False)]

    # Rename to standard column names
    df = df.rename(columns=COLUMN_RENAME)

    # Filter to equities only
    if equity_only and "asset_class" in df.columns:
        df = df[df["asset_class"] == "Equity"].copy()

    # Parse numeric columns
    for col in ["market_value", "weight_pct", "notional_value", "quantity", "price", "fx_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tag with as_of date
    df["as_of_date"] = as_of
    df = df.reset_index(drop=True)

    print(f"Holdings: {len(df):,} equities")
    return df


def save(df: pd.DataFrame, ticker: str = "ACWI", output_dir: str = "data") -> str:
    """Save holdings DataFrame to output_dir and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    as_of = df["as_of_date"].iloc[0].replace("-", "")
    fname = os.path.join(output_dir, f"constituents_{ticker}_{as_of}.csv")
    df.to_csv(fname, index=False)
    print(f"Saved {len(df):,} rows to {fname}")
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download iShares ETF holdings")
    parser.add_argument("--ticker", default="ACWI", help="Fund ticker (ACWI, URTH, EEM)")
    parser.add_argument("--all", action="store_true", help="Download all registered funds")
    args = parser.parse_args()

    tickers = list(FUNDS) if args.all else [args.ticker]
    for t in tickers:
        df = fetch(t)
        save(df, t, "data")
        print(f"\nSample ({t}):")
        print(df[["ticker", "name", "sector", "country", "weight_pct"]].head(10).to_string(index=False))
        print()
