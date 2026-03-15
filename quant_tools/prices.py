"""
Download total return price data via Yahoo Finance.

Uses adjusted close prices, which account for dividends and splits and
therefore proxy total returns in local currency.

Usage
-----
    from quant_tools.prices import fetch_prices

    # From a list of tickers
    prices = fetch_prices(["AAPL", "MSFT", "RR.L", "NOVO-B.CO"])

    # From an enriched constituent DataFrame (uses yahoo_ticker column)
    from quant_tools.constituents import fetch
    constituents = fetch("ACWI", enrich=True)
    prices = fetch_prices(constituents)

    # Monthly prices, last 5 years
    prices = fetch_prices(constituents, start="2020-01-01", freq="M")

    # Convert to returns
    returns = prices.pct_change().dropna(how="all")
"""

import os

import pandas as pd
import yfinance as yf

from ._defaults import _DEFAULT_DATA_DIR


def fetch_prices(
    tickers,
    start: str = "2015-01-01",
    end: str | None = None,
    freq: str = "M",
    batch_size: int = 500,
    output_dir: str | None = _DEFAULT_DATA_DIR,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers via Yahoo Finance.

    Parameters
    ----------
    tickers    : list/Series of Yahoo Finance tickers, OR a constituent
                 DataFrame with a 'yahoo_ticker' column
    start      : start date, e.g. '2015-01-01'
    end        : end date (default: today)
    freq       : 'D' for daily, 'M' for month-end, 'W' for week-end
    batch_size : tickers per yfinance request (default 500)
    output_dir : directory to save CSV (default: quant-tools/data/).
                 Pass None to skip saving.
    filename   : override the output filename (default: prices_{freq}.csv)

    Returns
    -------
    DataFrame of adjusted close prices.
    DatetimeIndex, columns = ticker symbols.
    Tickers with no data are dropped.
    """
    # Accept constituent DataFrame
    if isinstance(tickers, pd.DataFrame):
        if "yahoo_ticker" not in tickers.columns:
            raise ValueError("DataFrame must have a 'yahoo_ticker' column")
        ticker_list = tickers["yahoo_ticker"].dropna().unique().tolist()
    else:
        ticker_list = list(tickers)

    ticker_list = [t for t in ticker_list if t and str(t) != "nan"]
    n = len(ticker_list)
    print(f"Downloading prices for {n:,} tickers  (start={start}, freq={freq})...")

    # Download in batches
    frames = []
    n_batches = (n + batch_size - 1) // batch_size
    for i, start_idx in enumerate(range(0, n, batch_size)):
        batch = ticker_list[start_idx : start_idx + batch_size]
        print(f"  Batch {i+1}/{n_batches}  ({start_idx+1}–{min(start_idx+batch_size, n)} of {n})",
              end="  ")
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            print("no data")
            continue

        # Extract Close (auto_adjust=True means Close = adjusted close)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": batch[0]})

        frames.append(close)
        ok = close.notna().any().sum()
        print(f"matched {ok}/{len(batch)}")

    if not frames:
        raise RuntimeError("No price data returned for any ticker.")

    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index()

    # Resample to desired frequency
    if freq == "M":
        prices = prices.resample("ME").last()
    elif freq == "W":
        prices = prices.resample("W").last()
    # freq="D": keep as-is

    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    n_ok = prices.shape[1]
    print(f"\nPrices: {prices.shape[0]:,} periods × {n_ok:,} tickers "
          f"({n - n_ok:,} tickers had no data)")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname_default = f"prices_{freq.lower()}.csv"
        fname = os.path.join(output_dir, filename or fname_default)
        prices.to_csv(fname)
        print(f"Saved to {fname}")

    return prices
