"""
Download company fundamental data via Yahoo Finance.

Pulls fast_info (market data) + annual financial statements (income statement,
balance sheet, cash flow) and computes key derived ratios.

Usage
-----
    from quant_tools.fundamentals import fetch_fundamentals

    # From a list of tickers
    fund = fetch_fundamentals(["AAPL", "MSFT", "RR.L"])

    # From a constituent DataFrame
    from quant_tools.constituents import fetch
    constituents = fetch("ACWI", enrich=True)
    fund = fetch_fundamentals(constituents)

    # Resume an interrupted run (reads existing checkpoint file)
    fund = fetch_fundamentals(constituents, resume=True)

Columns returned
----------------
    Market data  : market_cap, shares, currency, fiscal_year_end
    Income stmt  : revenue, gross_profit, ebit, ebitda, net_income,
                   eps_diluted, rd_expense, da_expense
    Balance sheet: total_assets, total_debt, net_debt, book_value, cash
    Cash flow    : free_cash_flow, capex, buybacks
    Ratios       : pe_trailing, pb, ev, ev_ebitda, ev_revenue,
                   roe, net_margin, gross_margin, debt_equity, fcf_yield

Notes
-----
  Financial statement coverage is best for US stocks; many international
  tickers return partial or empty data. Ratios are set to NaN where inputs
  are missing or produce non-finite values (e.g. negative book value).
"""

import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from ._defaults import _DEFAULT_DATA_DIR

# Seconds to sleep between tickers to avoid rate limiting
_SLEEP = 0.3


def _safe_get(stmt: pd.DataFrame, *keys) -> float:
    """Return the most recent annual value for the first matching row key."""
    if stmt is None or stmt.empty:
        return np.nan
    for key in keys:
        if key in stmt.index:
            # Most recent column = leftmost after sorting columns descending
            cols = sorted(stmt.columns, reverse=True)
            for col in cols:
                v = stmt.loc[key, col]
                if pd.notna(v):
                    return float(v)
    return np.nan


def _fiscal_year_end(stmt: pd.DataFrame) -> str | None:
    if stmt is None or stmt.empty:
        return None
    cols = sorted(stmt.columns, reverse=True)
    return str(cols[0].date()) if hasattr(cols[0], "date") else str(cols[0])


def _fundamentals_one(ticker: str) -> dict:
    """Pull fundamentals for a single ticker. Returns a dict of metrics."""
    row: dict = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)

        # ── Market data ────────────────────────────────────────────────────────
        fi = t.fast_info
        market_cap = getattr(fi, "market_cap", None) or np.nan
        shares     = getattr(fi, "shares", None)     or np.nan
        currency   = getattr(fi, "currency", None)   or ""
        row.update({"market_cap": market_cap, "shares": shares, "currency": currency})

        # fast_info.market_cap for LSE stocks is in GBp (pence); financial
        # statements are in GBP — divide by 100 so ratios are consistent.
        market_cap_norm = market_cap / 100 if currency == "GBp" else market_cap

        # ── Financial statements ───────────────────────────────────────────────
        inc  = t.income_stmt
        bal  = t.balance_sheet
        cf   = t.cashflow

        row["fiscal_year_end"] = _fiscal_year_end(inc)

        # Income statement
        revenue      = _safe_get(inc, "Total Revenue", "Operating Revenue")
        gross_profit = _safe_get(inc, "Gross Profit")
        ebit         = _safe_get(inc, "EBIT")
        ebitda       = _safe_get(inc, "EBITDA", "Normalized EBITDA")
        net_income   = _safe_get(inc, "Net Income",
                                 "Net Income From Continuing Operation Net Minority Interest")
        eps_diluted  = _safe_get(inc, "Diluted EPS")
        rd_expense   = _safe_get(inc, "Research And Development")
        da_expense   = _safe_get(inc, "Reconciled Depreciation")

        row.update({
            "revenue": revenue, "gross_profit": gross_profit,
            "ebit": ebit, "ebitda": ebitda, "net_income": net_income,
            "eps_diluted": eps_diluted, "rd_expense": rd_expense,
            "da_expense": da_expense,
        })

        # Balance sheet
        total_assets = _safe_get(bal, "Total Assets")
        total_debt   = _safe_get(bal, "Total Debt", "Long Term Debt And Capital Lease Obligation")
        net_debt     = _safe_get(bal, "Net Debt")
        book_value   = _safe_get(bal, "Common Stock Equity", "Stockholders Equity",
                                 "Tangible Book Value")
        cash         = _safe_get(bal, "Cash And Cash Equivalents",
                                 "Cash Cash Equivalents And Short Term Investments")

        row.update({
            "total_assets": total_assets, "total_debt": total_debt,
            "net_debt": net_debt, "book_value": book_value, "cash": cash,
        })

        # Cash flow
        fcf     = _safe_get(cf, "Free Cash Flow")
        capex   = _safe_get(cf, "Capital Expenditure")
        buybacks = _safe_get(cf, "Repurchase Of Capital Stock")

        row.update({"free_cash_flow": fcf, "capex": capex, "buybacks": buybacks})

        # ── Derived ratios ─────────────────────────────────────────────────────
        def _ratio(num, den):
            try:
                v = num / den
                return float(v) if np.isfinite(v) else np.nan
            except Exception:
                return np.nan

        ev = market_cap_norm + net_debt if pd.notna(market_cap_norm) and pd.notna(net_debt) else np.nan

        row.update({
            "ev":           ev,
            "pe_trailing":  _ratio(market_cap_norm, net_income),
            "pb":           _ratio(market_cap_norm, book_value),
            "ev_ebitda":    _ratio(ev, ebitda),
            "ev_revenue":   _ratio(ev, revenue),
            "roe":          _ratio(net_income, book_value),
            "net_margin":   _ratio(net_income, revenue),
            "gross_margin": _ratio(gross_profit, revenue),
            "debt_equity":  _ratio(total_debt, book_value),
            "fcf_yield":    _ratio(fcf, market_cap_norm),
        })

    except Exception as e:
        row["_error"] = str(e)

    return row


def fetch_fundamentals(
    tickers,
    output_dir: str | None = _DEFAULT_DATA_DIR,
    filename: str | None = None,
    resume: bool = False,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """
    Download fundamental data for a list of tickers via Yahoo Finance.

    Parameters
    ----------
    tickers          : list/Series of Yahoo Finance tickers, OR a constituent
                       DataFrame with a 'yahoo_ticker' column
    output_dir       : directory to save results (default: quant-tools/data/).
                       Pass None to skip saving.
    filename         : override output filename (default: fundamentals.csv)
    resume           : if True, load existing output file and skip already-done
                       tickers (allows resuming an interrupted run)
    checkpoint_every : save progress to disk every N tickers (default 50)

    Returns
    -------
    DataFrame with one row per ticker and fundamental metrics as columns.
    """
    if isinstance(tickers, pd.DataFrame):
        if "yahoo_ticker" not in tickers.columns:
            raise ValueError("DataFrame must have a 'yahoo_ticker' column")
        ticker_list = tickers["yahoo_ticker"].dropna().unique().tolist()
    else:
        ticker_list = list(tickers)

    ticker_list = [str(t) for t in ticker_list if t and str(t) != "nan"]

    fname = os.path.join(output_dir or ".", filename or "fundamentals.csv")

    # ── Resume: load existing results and skip completed tickers ──────────────
    existing: pd.DataFrame | None = None
    done_tickers: set = set()
    if resume and output_dir and os.path.exists(fname):
        existing = pd.read_csv(fname, index_col=0)
        done_tickers = set(existing.index.tolist())
        print(f"Resuming — {len(done_tickers):,} tickers already done, "
              f"{len(ticker_list) - len(done_tickers):,} remaining")

    todo = [t for t in ticker_list if t not in done_tickers]
    n = len(todo)
    print(f"Downloading fundamentals for {n:,} tickers...")

    results: list[dict] = []
    for i, ticker in enumerate(todo):
        results.append(_fundamentals_one(ticker))

        if (i + 1) % 10 == 0 or (i + 1) == n:
            pct = (i + 1) / n * 100
            ok  = sum(1 for r in results if "_error" not in r)
            print(f"  {i+1:>5,}/{n:,}  ({pct:.0f}%)  matched {ok}/{i+1}", end="\r")

        # Checkpoint
        if output_dir and (i + 1) % checkpoint_every == 0:
            _save(results, existing, fname)

        time.sleep(_SLEEP)

    print()  # newline after \r progress

    df = _save(results, existing, fname if output_dir else None)

    ok = df["net_income"].notna().sum()
    print(f"\nFundamentals: {len(df):,} tickers  "
          f"({ok:,} with income statement data, {len(df)-ok:,} market-data only)")
    return df


def _save(
    results: list[dict],
    existing: pd.DataFrame | None,
    fname: str | None,
) -> pd.DataFrame:
    """Merge new results with any existing data and optionally save."""
    new_df = pd.DataFrame(results).set_index("ticker") if results else pd.DataFrame()

    if existing is not None and not existing.empty:
        df = pd.concat([existing, new_df])
    else:
        df = new_df

    df = df[~df.index.duplicated(keep="last")]

    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_csv(fname)

    return df
