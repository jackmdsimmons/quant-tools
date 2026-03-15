"""
Security identifier enrichment for equity constituent DataFrames.

Two independent functions:

  add_yahoo_ticker(df)   — instant lookup, no API, adds yahoo_ticker column
  enrich_with_figi(df)   — calls OpenFIGI API, adds figi / composite_figi /
                           share_class_figi columns

OpenFIGI API limits (free tier, no API key)
-------------------------------------------
  - 10 items per request
  - 10 requests per minute
  Set env var OPENFIGI_API_KEY to increase to 100 items / 25 req per min.

Usage
-----
    from quant_tools.identifiers import add_yahoo_ticker, enrich_with_figi
    from quant_tools.constituents import fetch

    df = fetch("ACWI")
    df = add_yahoo_ticker(df)           # adds yahoo_ticker, fast
    df = enrich_with_figi(df)           # adds FIGI ids, ~3 min for ACWI
"""

import os
import time
import requests
import pandas as pd

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

# Batch and rate-limit settings
_BATCH_FREE  = 10    # items per request, no API key
_BATCH_KEY   = 100   # items per request, with API key
_SLEEP_FREE  = 6.5   # seconds between batches, no key (10 req/min)
_SLEEP_KEY   = 2.6   # seconds between batches, with key (25 req/min)

# Security types accepted as equity
_EQUITY_TYPES = {
    "Common Stock", "ETP", "ADR", "GDR", "REIT",
    "Preference", "Ltd Part", "MLP", "Stapled Security",
}

# ── Ticker normalisation for OpenFIGI ─────────────────────────────────────────

# iShares strips dots from US tickers (BRK.B → BRKB). Map back to real tickers.
_TICKER_ALIASES: dict[str, str] = {
    "BRKB": "BRK.B",
    "BRKA": "BRK.A",
}

# LSE tickers end with "." in iShares (BP., RR., BA., NG.).
# OpenFIGI expects the Bloomberg convention: trailing "." → "/"
_LSE_EXCHANGES = {"London Stock Exchange"}

# Nordic Stockholm/Copenhagen: iShares includes share class as "TICKER CLASS"
# (e.g. "INVE B", "NOVO B"). OpenFIGI wants spaces removed (INVEB, NOVOB).
_NORDIC_CONCAT_EXCHANGES = {"Nasdaq Omx Nordic", "Omx Nordic Exchange Copenhagen A/S"}

# Nordic Helsinki: iShares appends country code ("NDA FI"). OpenFIGI uses
# only the base ticker before the space (NDA).
_NORDIC_PREFIX_EXCHANGES = {"Nasdaq Omx Helsinki Ltd."}

# Italian stocks: micCode XMIL doesn't resolve in OpenFIGI; exchCode "IM" does.
_EXCHCODE_OVERRIDE: dict[str, str] = {
    "Borsa Italiana": "IM",
}


def _normalize_figi_ticker(ticker: str, exchange: str) -> str:
    """Apply exchange-specific ticker normalisation before OpenFIGI lookup."""
    ticker = _TICKER_ALIASES.get(ticker, ticker)
    if exchange in _LSE_EXCHANGES:
        if ticker.endswith("."):
            ticker = ticker[:-1] + "/"
    elif exchange in _NORDIC_CONCAT_EXCHANGES:
        ticker = ticker.replace(" ", "")
    elif exchange in _NORDIC_PREFIX_EXCHANGES:
        ticker = ticker.split()[0]
    return ticker

# ── Exchange name → Yahoo Finance ticker suffix ───────────────────────────────
# Source: iShares ACWI 'exchange' column values mapped to Yahoo Finance suffixes.
# US listings have no suffix. Unknown exchanges fall back to "".
EXCHANGE_TO_YAHOO_SUFFIX: dict[str, str] = {
    # United States (no suffix)
    "NASDAQ":                                      "",
    "NYSE":                                        "",
    "Cboe BZX":                                    "",
    "NYSE Arca":                                   "",

    # Asia Pacific
    "Tokyo Stock Exchange":                        ".T",
    "Hong Kong Exchanges And Clearing Ltd":        ".HK",
    "Taiwan Stock Exchange":                       ".TW",
    "Gretai Securities Market":                    ".TWO",
    "Korea Exchange (Stock Market)":               ".KS",
    "Korea Exchange (Kosdaq)":                     ".KQ",
    "Asx - All Markets":                           ".AX",
    "New Zealand Exchange Ltd":                    ".NZ",
    "Singapore Exchange":                          ".SI",
    "National Stock Exchange Of India":            ".NS",
    "Bse Ltd":                                     ".BO",
    "Shanghai Stock Exchange":                     ".SS",
    "Shenzhen Stock Exchange":                     ".SZ",
    "Bursa Malaysia":                              ".KL",
    "Stock Exchange Of Thailand":                  ".BK",
    "Indonesia Stock Exchange":                    ".JK",
    "Philippine Stock Exchange Inc.":              ".PS",

    # Europe
    "London Stock Exchange":                       ".L",
    "Xetra":                                       ".DE",
    "SIX Swiss Exchange":                          ".SW",
    "Nyse Euronext - Euronext Paris":              ".PA",
    "Euronext Amsterdam":                          ".AS",
    "Borsa Italiana":                              ".MI",
    "Bolsa De Madrid":                             ".MC",
    "Nasdaq Omx Nordic":                           ".ST",
    "Nasdaq Omx Helsinki Ltd.":                    ".HE",
    "Omx Nordic Exchange Copenhagen A/S":          ".CO",
    "Oslo Bors Asa":                               ".OL",
    "Nyse Euronext - Euronext Brussels":           ".BR",
    "Nyse Euronext - Euronext Lisbon":             ".LS",
    "Irish Stock Exchange - All Market":           ".IR",
    "Warsaw Stock Exchange/Equities/Main Market":  ".WA",
    "Prague Stock Exchange":                       ".PR",
    "Budapest Stock Exchange":                     ".BD",
    "Athens Exchange S.A. Cash Market":            ".AT",
    "Wiener Boerse Ag":                            ".VI",
    "Istanbul Stock Exchange":                     ".IS",

    # Americas
    "Toronto Stock Exchange":                      ".TO",
    "XBSP":                                        ".SA",
    "Bolsa Mexicana De Valores":                   ".MX",
    "Santiago Stock Exchange":                     ".SN",
    "Bolsa De Valores De Colombia":                ".CL",

    # Middle East / Africa
    "Johannesburg Stock Exchange":                 ".JO",
    "Saudi Stock Exchange":                        ".SR",
    "Tel Aviv Stock Exchange":                     ".TA",
    "Dubai Financial Market":                      ".DU",
    "Abu Dhabi Securities Exchange":               ".AD",
    "Qatar Exchange":                              ".QA",
    "Kuwait Stock Exchange":                       "",    # poor Yahoo coverage
    "Egyptian Exchange":                           ".CA",
}

# ── Exchange name → ISO 10383 MIC code (for OpenFIGI micCode parameter) ───────
EXCHANGE_TO_MIC: dict[str, str | None] = {
    # US: use exchCode="US" instead of micCode (micCode=XNAS returns no result)
    "NASDAQ":                                      None,
    "NYSE":                                        None,
    "Cboe BZX":                                    None,
    "NYSE Arca":                                   None,

    # Asia Pacific
    "Tokyo Stock Exchange":                        "XTKS",
    "Hong Kong Exchanges And Clearing Ltd":        "XHKG",
    "Taiwan Stock Exchange":                       "XTAI",
    "Gretai Securities Market":                    "ROCO",
    "Korea Exchange (Stock Market)":               "XKRX",
    "Korea Exchange (Kosdaq)":                     "XKOS",
    "Asx - All Markets":                           "XASX",
    "New Zealand Exchange Ltd":                    "XNZE",
    "Singapore Exchange":                          "XSES",
    "National Stock Exchange Of India":            "XNSE",
    "Bse Ltd":                                     "XBOM",
    "Shanghai Stock Exchange":                     "XSHG",
    "Shenzhen Stock Exchange":                     "XSHE",
    "Bursa Malaysia":                              "XKLS",
    "Stock Exchange Of Thailand":                  "XBKK",
    "Indonesia Stock Exchange":                    "XIDX",
    "Philippine Stock Exchange Inc.":              "XPHS",

    # Europe
    "London Stock Exchange":                       "XLON",
    "Xetra":                                       "XETR",
    "SIX Swiss Exchange":                          "XSWX",
    "Nyse Euronext - Euronext Paris":              "XPAR",
    "Euronext Amsterdam":                          "XAMS",
    "Borsa Italiana":                              "XMIL",
    "Bolsa De Madrid":                             "XMAD",
    "Nasdaq Omx Nordic":                           "XSTO",
    "Nasdaq Omx Helsinki Ltd.":                    "XHEL",
    "Omx Nordic Exchange Copenhagen A/S":          "XCSE",
    "Oslo Bors Asa":                               "XOSL",
    "Nyse Euronext - Euronext Brussels":           "XBRU",
    "Nyse Euronext - Euronext Lisbon":             "XLIS",
    "Irish Stock Exchange - All Market":           "XDUB",
    "Warsaw Stock Exchange/Equities/Main Market":  "XWAR",
    "Prague Stock Exchange":                       "XPRA",
    "Budapest Stock Exchange":                     "XBUD",
    "Athens Exchange S.A. Cash Market":            "ASEX",
    "Wiener Boerse Ag":                            "XWBO",
    "Istanbul Stock Exchange":                     "XIST",

    # Americas
    "Toronto Stock Exchange":                      "XTSE",
    "XBSP":                                        "BVMF",
    "Bolsa Mexicana De Valores":                   "XMEX",
    "Santiago Stock Exchange":                     "XSGO",
    "Bolsa De Valores De Colombia":                "XBOG",

    # Middle East / Africa
    "Johannesburg Stock Exchange":                 "XJSE",
    "Saudi Stock Exchange":                        "XSAU",
    "Tel Aviv Stock Exchange":                     "XTAE",
    "Dubai Financial Market":                      "XDFM",
    "Abu Dhabi Securities Exchange":               "XADS",
    "Qatar Exchange":                              "XQAT",
    "Kuwait Stock Exchange":                       "XKUW",
    "Egyptian Exchange":                           "XCAI",
}

_US_EXCHANGES = {k for k, v in EXCHANGE_TO_MIC.items() if v is None}


def add_yahoo_ticker(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    exchange_col: str = "exchange",
) -> pd.DataFrame:
    """
    Add a yahoo_ticker column using a local exchange-name lookup table.
    No API calls — instant.

    Yahoo ticker = local ticker + exchange suffix (e.g. '7203' + '.T' = '7203.T').
    US listings have no suffix. Unknown exchanges fall back to the raw ticker.

    Parameters
    ----------
    df           : DataFrame with ticker and exchange columns
    ticker_col   : column containing the local exchange ticker
    exchange_col : column containing the iShares exchange name

    Returns
    -------
    DataFrame with added 'yahoo_ticker' column
    """
    def _make_ticker(row):
        suffix = EXCHANGE_TO_YAHOO_SUFFIX.get(row[exchange_col], "")
        ticker = _TICKER_ALIASES.get(str(row[ticker_col]), str(row[ticker_col]))
        ticker = ticker.rstrip(".").replace(".", "-").replace(" ", "-")  # Yahoo uses hyphens
        return f"{ticker}{suffix}"

    out = df.copy()
    out["yahoo_ticker"] = df.apply(_make_ticker, axis=1)
    return out


def enrich_with_figi(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    exchange_col: str = "exchange",
    api_key: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add FIGI identifiers by querying the OpenFIGI API.

    Uses micCode for non-US exchanges to target primary listings and avoid
    defaulting to ADRs. US exchanges use exchCode='US'.

    Parameters
    ----------
    df           : DataFrame with ticker and exchange columns
    ticker_col   : column containing the local exchange ticker
    exchange_col : column containing the iShares exchange name
    api_key      : OpenFIGI API key (optional). Set OPENFIGI_API_KEY env var
                   as an alternative. Increases batch size to 100 and rate
                   limit to 25 req/min.
    verbose      : print progress

    Returns
    -------
    DataFrame with added columns: figi, composite_figi, share_class_figi
    """
    key = api_key or os.environ.get("OPENFIGI_API_KEY")
    batch_size = _BATCH_KEY if key else _BATCH_FREE
    sleep_s    = _SLEEP_KEY if key else _SLEEP_FREE

    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-OPENFIGI-APIKEY"] = key

    tickers   = df[ticker_col].tolist()
    exchanges = df[exchange_col].tolist()
    n         = len(df)

    raw_results = [None] * n
    n_batches   = (n + batch_size - 1) // batch_size

    for batch_num, batch_start in enumerate(range(0, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        idx       = list(range(batch_start, batch_end))

        if verbose:
            print(f"  Batch {batch_num + 1}/{n_batches}  "
                  f"({batch_start + 1}–{batch_end} of {n})", end="  ")

        reqs = []
        for i in idx:
            exchange = exchanges[i]
            ticker   = _normalize_figi_ticker(str(tickers[i]), exchange)
            if exchange in _US_EXCHANGES:
                reqs.append({"idType": "TICKER", "idValue": ticker, "exchCode": "US"})
            elif exchange in _EXCHCODE_OVERRIDE:
                reqs.append({"idType": "TICKER", "idValue": ticker, "exchCode": _EXCHCODE_OVERRIDE[exchange]})
            else:
                mic = EXCHANGE_TO_MIC.get(exchange)
                req = {"idType": "TICKER", "idValue": ticker}
                if mic:
                    req["micCode"] = mic
                reqs.append(req)

        try:
            resp = requests.post(OPENFIGI_URL, json=reqs, headers=headers, timeout=30)
            resp.raise_for_status()
            batch_data = resp.json()
            if verbose:
                hits = sum(1 for r in batch_data if r.get("data"))
                print(f"matched {hits}/{len(reqs)}")
        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")
            if batch_end < n:
                time.sleep(sleep_s)
            continue

        for i, result in zip(idx, batch_data):
            data = result.get("data", [])
            equity = [r for r in data if r.get("marketSector") == "Equity"
                      and r.get("securityType") in _EQUITY_TYPES]
            raw_results[i] = equity[0] if equity else (data[0] if data else None)

        if batch_end < n:
            time.sleep(sleep_s)

    # Build output columns
    figi_out, comp_out, sc_out = [], [], []
    for match in raw_results:
        if match:
            figi_out.append(match.get("figi"))
            comp_out.append(match.get("compositeFIGI"))
            sc_out.append(match.get("shareClassFIGI"))
        else:
            figi_out.append(None)
            comp_out.append(None)
            sc_out.append(None)

    out = df.copy()
    out["figi"]             = figi_out
    out["composite_figi"]   = comp_out
    out["share_class_figi"] = sc_out

    if verbose:
        mapped = sum(1 for f in figi_out if f)
        print(f"\nFIGI mapped: {mapped:,} / {n:,} ({mapped/n:.1%})")

    return out
