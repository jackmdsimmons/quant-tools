"""
Download total return price data via Yahoo Finance.

Uses adjusted close prices, which account for dividends and splits and
therefore proxy total returns in local currency.

Usage
-----
    from quant_tools.prices import fetch_prices

    # From a list of tickers
    prices = fetch_prices(["AAPL", "MSFT", "RR.L", "NOVO-B.CO"])

    # From an enriched constituent DataFrame — also saves USD version
    from quant_tools.constituents import fetch
    constituents = fetch("ACWI", enrich=True)
    prices, prices_usd = fetch_prices(constituents, to_usd=True)

    # Monthly prices, last 5 years, local currency only
    prices = fetch_prices(constituents, start="2020-01-01", freq="M")

    # Convert to returns
    returns = prices.pct_change().dropna(how="all")

Notes
-----
  - LSE prices are quoted in pence on Yahoo Finance; the USD conversion
    divides by 100 automatically for GBp-denominated tickers.
  - to_usd=True requires the input to be a constituent DataFrame with a
    'currency' column, or a ticker→currency mapping dict.
"""

import os

import pandas as pd
import yfinance as yf

from ._defaults import _DEFAULT_DATA_DIR

# Yahoo Finance FX ticker format: {CCY}USD=X  (e.g. GBPUSD=X)
# Special case: LSE quotes prices in GBp (pence), not GBP — divide by 100.
_PENCE_CURRENCIES = {"GBp"}

# Currencies that are already USD — no conversion needed
_USD_CURRENCIES = {"USD", "USX"}


def _fetch_fx_rates(
    currencies: set[str],
    start: str,
    end: str | None,
    freq: str,
) -> pd.DataFrame:
    """Download USD FX rates for a set of currency codes."""
    fx_tickers = []
    for ccy in currencies:
        if ccy in _USD_CURRENCIES or ccy in _PENCE_CURRENCIES:
            continue
        fx_tickers.append(f"{ccy}USD=X")

    if not fx_tickers:
        return pd.DataFrame()

    print(f"Downloading FX rates for {len(fx_tickers)} currencies: "
          f"{[t.replace('USD=X','') for t in fx_tickers]}")

    raw = yf.download(fx_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    fx = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": fx_tickers[0]})

    if freq == "M":
        fx = fx.resample("ME").last()
    elif freq == "W":
        fx = fx.resample("W").last()

    # Rename GBPUSD=X → GBP etc.
    fx.columns = [c.replace("USD=X", "") for c in fx.columns]
    fx = fx.ffill()
    return fx


def _to_usd_prices(
    prices: pd.DataFrame,
    ticker_currency: dict[str, str],
    fx: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a local-currency price DataFrame to USD."""
    usd_frames = {}
    for ticker in prices.columns:
        ccy = ticker_currency.get(ticker, "USD")
        series = prices[ticker].copy()

        if ccy in _USD_CURRENCIES:
            usd_frames[ticker] = series
        elif ccy in _PENCE_CURRENCIES:
            # GBp → GBP → USD
            gbp_usd = fx.get("GBP")
            if gbp_usd is not None:
                usd_frames[ticker] = series / 100 * gbp_usd.reindex(series.index, method="ffill")
            else:
                usd_frames[ticker] = series / 100  # best effort
        else:
            rate = fx.get(ccy)
            if rate is not None:
                usd_frames[ticker] = series * rate.reindex(series.index, method="ffill")
            else:
                usd_frames[ticker] = series  # no rate available, leave as-is

    return pd.DataFrame(usd_frames)


def fetch_prices(
    tickers,
    start: str = "2015-01-01",
    end: str | None = None,
    freq: str = "M",
    to_usd: bool = False,
    batch_size: int = 500,
    output_dir: str | None = _DEFAULT_DATA_DIR,
    filename: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices for a list of tickers via Yahoo Finance.

    Parameters
    ----------
    tickers    : list/Series of Yahoo Finance tickers, OR a constituent
                 DataFrame with 'yahoo_ticker' (and 'currency' for to_usd)
    start      : start date, e.g. '2015-01-01'
    end        : end date (default: today)
    freq       : 'D' for daily, 'M' for month-end, 'W' for week-end
    to_usd     : if True, also return USD-converted prices. Requires tickers
                 to be a constituent DataFrame with a 'currency' column.
    batch_size : tickers per yfinance request (default 500)
    output_dir : directory to save CSV(s) (default: quant-tools/data/).
                 Pass None to skip saving.
    filename   : override output filename for local-currency file

    Returns
    -------
    If to_usd=False : DataFrame of local-currency adjusted close prices
    If to_usd=True  : tuple of (local_prices, usd_prices)

    In both cases columns = ticker symbols, index = DatetimeIndex.
    Tickers with no data are dropped.
    """
    # Parse input — build ticker list and optional currency map
    ticker_currency: dict[str, str] = {}
    if isinstance(tickers, pd.DataFrame):
        if "yahoo_ticker" not in tickers.columns:
            raise ValueError("DataFrame must have a 'yahoo_ticker' column")
        if to_usd and "currency" not in tickers.columns:
            raise ValueError("to_usd=True requires a 'currency' column in the DataFrame")
        rows = tickers.dropna(subset=["yahoo_ticker"])
        ticker_list = rows["yahoo_ticker"].unique().tolist()
        if to_usd:
            ticker_currency = dict(zip(rows["yahoo_ticker"], rows["currency"]))
    else:
        if to_usd:
            raise ValueError("to_usd=True requires a constituent DataFrame with a 'currency' column")
        ticker_list = list(tickers)

    ticker_list = [str(t) for t in ticker_list if t and str(t) != "nan"]
    n = len(ticker_list)
    print(f"Downloading prices for {n:,} tickers  (start={start}, freq={freq})...")

    # ── Download prices in batches ─────────────────────────────────────────────
    frames = []
    n_batches = (n + batch_size - 1) // batch_size
    for i, start_idx in enumerate(range(0, n, batch_size)):
        batch = ticker_list[start_idx : start_idx + batch_size]
        print(f"  Batch {i+1}/{n_batches}  "
              f"({start_idx+1}–{min(start_idx+batch_size, n)} of {n})", end="  ")
        raw = yf.download(
            batch, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )
        if raw.empty:
            print("no data")
            continue

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) \
            else raw[["Close"]].rename(columns={"Close": batch[0]})

        frames.append(close)
        print(f"matched {close.notna().any().sum()}/{len(batch)}")

    if not frames:
        raise RuntimeError("No price data returned for any ticker.")

    prices = pd.concat(frames, axis=1).sort_index()

    # Resample
    if freq == "M":
        prices = prices.resample("ME").last()
    elif freq == "W":
        prices = prices.resample("W").last()

    prices = prices.dropna(axis=1, how="all")
    n_ok = prices.shape[1]
    print(f"\nPrices: {prices.shape[0]:,} periods × {n_ok:,} tickers "
          f"({n - n_ok:,} had no data)")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # ── Save local currency ────────────────────────────────────────────────────
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, filename or f"prices_{freq.lower()}.csv")
        prices.to_csv(fname)
        print(f"Saved local-currency prices to {fname}")

    if not to_usd:
        return prices

    # ── USD conversion ─────────────────────────────────────────────────────────
    currencies = {c for c in ticker_currency.values() if c and str(c) != "nan"}
    fx = _fetch_fx_rates(currencies, start=start, end=end, freq=freq)

    prices_usd = _to_usd_prices(prices, ticker_currency, fx)
    prices_usd = prices_usd.dropna(axis=1, how="all")

    print(f"USD prices: {prices_usd.shape[0]:,} periods × {prices_usd.shape[1]:,} tickers")

    if output_dir is not None:
        fname_usd = os.path.join(output_dir, f"prices_{freq.lower()}_usd.csv")
        prices_usd.to_csv(fname_usd)
        print(f"Saved USD prices to {fname_usd}")

    return prices, prices_usd
