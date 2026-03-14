"""
Fetch academic benchmark factor data.

Sources
-------
  Kenneth French Data Library  — FF3, FF5, Momentum (monthly + daily)
  AQR Data Library             — Betting Against Beta (BAB), Quality Minus Junk (QMJ)

All returns are returned as decimals (not percentages).

Usage
-----
    from quant_tools.benchmarks import fetch_french, fetch_aqr, FRENCH_DATASETS

    # French factors
    ff3  = fetch_french("FF3")       # Mkt-RF, SMB, HML, RF — monthly from 1926
    ff5  = fetch_french("FF5")       # + RMW, CMA — monthly from 1963
    mom  = fetch_french("MOM")       # Mom — monthly from 1927
    ff3d = fetch_french("FF3_daily") # Mkt-RF, SMB, HML, RF — daily from 1926

    # AQR factors (returns dict of {region: DataFrame})
    bab = fetch_aqr("BAB")          # Betting Against Beta by region
    qmj = fetch_aqr("QMJ")          # Quality Minus Junk by region

    # Convenience: get US factors only
    bab_us = fetch_aqr("BAB")["USA"]
"""

import io
import zipfile

import pandas as pd
import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FRENCH_DATASETS = {
    "FF3":       {"zip": "F-F_Research_Data_Factors_CSV.zip",           "skiprows": 3,  "freq": "M"},
    "FF5":       {"zip": "F-F_Research_Data_5_Factors_2x3_CSV.zip",     "skiprows": 3,  "freq": "M"},
    "MOM":       {"zip": "F-F_Momentum_Factor_CSV.zip",                 "skiprows": 13, "freq": "M"},
    "FF3_daily": {"zip": "F-F_Research_Data_Factors_daily_CSV.zip",     "skiprows": 3,  "freq": "D"},
}

_AQR_BASE = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"

AQR_DATASETS = {
    "BAB": "Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "QMJ": "Quality-Minus-Junk-Factors-Monthly.xlsx",
}


# ── French ────────────────────────────────────────────────────────────────────

def fetch_french(dataset: str = "FF3") -> pd.DataFrame:
    """
    Download a Kenneth French factor dataset.

    Parameters
    ----------
    dataset : one of FRENCH_DATASETS keys — 'FF3', 'FF5', 'MOM', 'FF3_daily'

    Returns
    -------
    DataFrame with DatetimeIndex (month-end for monthly, daily otherwise).
    All values in decimal form (divided by 100).
    Columns: as named in the source CSV (e.g. Mkt-RF, SMB, HML, RF).
    """
    if dataset not in FRENCH_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(FRENCH_DATASETS)}")

    spec = FRENCH_DATASETS[dataset]
    url  = _FRENCH_BASE + spec["zip"]

    print(f"Downloading French {dataset}...")
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()

    # Unzip in memory — French ZIPs contain a single CSV
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    df = pd.read_csv(
        io.StringIO(raw),
        skiprows=spec["skiprows"],
        index_col=0,
        na_values=["-99.99", "-999"],
    )
    df.columns = df.columns.str.strip()
    df.index   = df.index.astype(str).str.strip()

    # French files have both monthly and annual sections separated by blank rows.
    # Keep only rows whose index looks like a valid date (6-digit YYYYMM or 8-digit YYYYMMDD).
    if spec["freq"] == "M":
        mask = df.index.str.match(r"^\d{6}$")
        df   = df[mask].copy()
        df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        mask = df.index.str.match(r"^\d{8}$")
        df   = df[mask].copy()
        df.index = pd.to_datetime(df.index, format="%Y%m%d")

    df = df.apply(pd.to_numeric, errors="coerce") / 100
    df = df.dropna(how="all")
    df.index.name = "date"

    print(f"  {len(df):,} observations  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ── AQR ───────────────────────────────────────────────────────────────────────

def _parse_aqr_sheet(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    AQR Excel sheets have variable-length description headers before the data.
    Find the first row where column A looks like a date and use that as the start.
    """
    if raw_df.empty or raw_df.shape[1] == 0:
        return pd.DataFrame()

    # Find the column-header row — AQR files have 'DATE' in column A
    header_row = None
    for i, val in enumerate(raw_df.iloc[:, 0]):
        if str(val).strip().upper() == "DATE":
            header_row = i
            break

    if header_row is None:
        return pd.DataFrame()

    cols = [str(c).strip() for c in raw_df.iloc[header_row].tolist()]
    data = raw_df.iloc[header_row + 1:].copy()
    data.columns = cols
    data = data.rename(columns={"DATE": "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all")
    data.index = data.index + pd.offsets.MonthEnd(0)
    return data


def fetch_aqr(dataset: str = "BAB") -> dict[str, pd.DataFrame]:
    """
    Download an AQR factor dataset.

    Parameters
    ----------
    dataset : one of AQR_DATASETS keys — 'BAB', 'QMJ'

    Returns
    -------
    Dict mapping sheet/region name to DataFrame with DatetimeIndex (month-end).
    All values in decimal form.

    Common keys include 'USA', 'Global', 'Global ex USA'.
    Access US data with: fetch_aqr('BAB')['USA']
    """
    if dataset not in AQR_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(AQR_DATASETS)}")

    url = _AQR_BASE + AQR_DATASETS[dataset]
    print(f"Downloading AQR {dataset}...")
    resp = requests.get(url, headers=_HEADERS, timeout=120)
    resp.raise_for_status()

    xls    = pd.ExcelFile(io.BytesIO(resp.content))
    sheets = xls.sheet_names
    print(f"  Sheets: {sheets}")

    result = {}
    for sheet in sheets:
        try:
            raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=sheet, header=None)
            df  = _parse_aqr_sheet(raw)
            if not df.empty:
                result[sheet] = df
                print(f"  {sheet}: {len(df):,} obs  "
                      f"({df.index[0].date()} to {df.index[-1].date()})")
        except Exception:
            pass  # skip sheets that don't contain time-series data

    return result
