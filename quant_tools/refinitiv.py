"""
LSEG Data (Refinitiv) Platform API client.

Requires the `lseg-data` package and a Platform session config file.

Installation
------------
    pip install lseg-data

Config file  (~/.lseg-data/lseg-data.config.json)
--------------------------------------------------
    {
        "sessions": {
            "default": "platform.ldp",
            "platform": {
                "ldp": {
                    "app-key":  "YOUR_APP_KEY",
                    "username": "YOUR_MACHINE_ID",
                    "password": "YOUR_PASSWORD"
                }
            }
        }
    }

Usage
-----
    from quant_tools.refinitiv import RefinitivClient

    with RefinitivClient() as rl:
        prices = rl.fetch_prices(["AAPL.O", "BP.L"], start="2020-01-01")
        fund   = rl.fetch_fundamentals(["AAPL.O", "BP.L"])
        est    = rl.fetch_estimates(["AAPL.O", "BP.L"])
        esg    = rl.fetch_esg(["AAPL.O", "BP.L"])

RIC format
----------
    US equities  : AAPL.O (NASDAQ), MSFT.O, JPM.N (NYSE)
    LSE          : BP.L, SHEL.L
    Euronext     : AIRP.PA, ASML.AS
    Deutsche     : VOWG_p.F
    OMX          : VOLV-B.ST
    Japan        : 7203.T
    Pass a constituent DataFrame with 'ric' column, or list of RIC strings.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import pandas as pd

# ── Lazy import so module loads even without lseg-data installed ──────────────
def _ld():
    try:
        import lseg.data as ld
        return ld
    except ImportError:
        raise ImportError(
            "lseg-data is not installed. Run:\n"
            "    pip install lseg-data"
        )


# ── Field definitions ─────────────────────────────────────────────────────────

# Historical price / volume fields (get_history)
_PRICE_FIELDS = [
    "TR.PriceClose",
    "TR.PriceOpen",
    "TR.PriceHigh",
    "TR.PriceLow",
    "TR.Volume",
    "TR.VWAP",
]

# WorldScope annual fundamentals (get_data with period params)
_FUNDAMENTAL_FIELDS = {
    # Income statement
    "revenue":       "TR.Revenue",
    "gross_profit":  "TR.GrossProfit",
    "ebitda":        "TR.EBITDA",
    "ebit":          "TR.EBIT",
    "net_income":    "TR.NetIncome",
    "eps_diluted":   "TR.EPSDiluted",
    "rd_expense":    "TR.ResearchDevelopmentExpense",
    "da_expense":    "TR.DepreciationAmortization",
    # Balance sheet
    "total_assets":  "TR.TotalAssetsReported",
    "total_debt":    "TR.TotalDebt",
    "net_debt":      "TR.NetDebt",
    "book_value":    "TR.BookValuePerShare",
    "cash":          "TR.CashAndSTInvestments",
    # Cash flow
    "free_cash_flow":"TR.FreeCashFlow",
    "capex":         "TR.CapitalExpenditures",
    "buybacks":      "TR.ShareBuyback",
    # Market / valuation
    "market_cap":    "TR.CompanyMarketCap",
    "shares":        "TR.SharesOutstanding",
    "pe_trailing":   "TR.PriceToEarningsRatio",
    "pb":            "TR.PriceToBVPerShare",
    "ev":            "TR.EnterpriseValue",
    "ev_ebitda":     "TR.EVToEBITDA",
    "ev_revenue":    "TR.EVToSales",
    "roe":           "TR.ReturnOnEquity",
    "net_margin":    "TR.NetProfitMarginActValue",
    "gross_margin":  "TR.GrossProfitMarginActValue",
    "debt_equity":   "TR.DebtToTotalEquity",
    "fcf_yield":     "TR.FCFYield",
}

# IBES / StarMine consensus estimates
_ESTIMATE_FIELDS = {
    # EPS
    "eps_est_fy1":      "TR.EPSMeanEstimate",          # FY+1 consensus EPS
    "eps_est_fy2":      "TR.EPSMeanEstimate",          # FY+2 (different period param)
    "eps_smart":        "TR.EPSSmartEstimate",         # StarMine SmartEstimate
    "eps_surprise":     "TR.EPSSurprise",
    "eps_revision":     "TR.EPSRevisionsSurprise",
    # Revenue
    "rev_est_fy1":      "TR.RevenueMeanEstimate",
    "rev_est_fy2":      "TR.RevenueMeanEstimate",
    "rev_surprise":     "TR.RevenueSurprise",
    # EBITDA
    "ebitda_est_fy1":   "TR.EBITDAMeanEstimate",
    # Other
    "num_analysts":     "TR.NumberOfEstimates",
    "rec_mean":         "TR.TPMeanRec",                # broker recommendation mean
    "target_price":     "TR.TPMean",
}

# ESG scores (Refinitiv ESG)
_ESG_FIELDS = {
    "esg_score":         "TR.TRESGScore",
    "esg_combined":      "TR.TRESGCScore",             # score + controversies
    "env_score":         "TR.EnvironmentPillarScore",
    "social_score":      "TR.SocialPillarScore",
    "gov_score":         "TR.GovernancePillarScore",
    "esg_controversy":   "TR.TRESGControversiesScore",
    "esg_year":          "TR.TRESGScore.periodenddate",
}


# ── Client ────────────────────────────────────────────────────────────────────

class RefinitivClient:
    """
    Context-manager wrapper around an LSEG Data Platform session.

    Parameters
    ----------
    config_path : path to lseg-data config JSON, or None to use default
                  (~/.lseg-data/lseg-data.config.json)
    """

    def __init__(self, config_path: str | None = None):
        self._config_path = config_path
        self._ld = _ld()

    def __enter__(self):
        kwargs = {}
        if self._config_path:
            kwargs["config_name"] = self._config_path
        self._ld.open_session("platform", **kwargs)
        print("LSEG Data session opened.")
        return self

    def __exit__(self, *_):
        self._ld.close_session()
        print("LSEG Data session closed.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ric_list(tickers) -> list[str]:
        """Accept a list, Series, or constituent DataFrame with 'ric' column."""
        if isinstance(tickers, pd.DataFrame):
            col = next((c for c in ["ric", "RIC", "yahoo_ticker"] if c in tickers.columns), None)
            if col is None:
                raise ValueError("DataFrame must have a 'ric' column")
            return tickers[col].dropna().unique().tolist()
        return [str(t) for t in tickers if t and str(t) != "nan"]

    @staticmethod
    def _chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # ── Pricing ───────────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        tickers,
        start: str = "2015-01-01",
        end: str | None = None,
        fields: list[str] | None = None,
        freq: str = "monthly",
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price history.

        Parameters
        ----------
        tickers  : list of RICs or constituent DataFrame
        start    : start date 'YYYY-MM-DD'
        end      : end date (default: today)
        fields   : override default fields (default: close + volume)
        freq     : 'daily' | 'weekly' | 'monthly'
        output_dir : save CSV here if provided

        Returns
        -------
        MultiIndex DataFrame: (date) × (field, ric) or flat close-price DataFrame
        if only TR.PriceClose requested.
        """
        ld = self._ld
        rics = self._ric_list(tickers)
        fields = fields or ["TR.PriceClose", "TR.Volume"]

        interval_map = {"daily": "1D", "weekly": "1W", "monthly": "1M"}
        interval = interval_map.get(freq, "1M")

        print(f"Fetching {freq} prices for {len(rics):,} RICs ({start} → {end or 'today'})...")

        frames = []
        for batch in self._chunk(rics, 100):
            df = ld.get_history(
                universe=batch,
                fields=fields,
                interval=interval,
                start=start,
                end=end,
            )
            frames.append(df)

        prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        print(f"  {prices.shape[0]:,} periods × {len(rics):,} RICs")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f"rl_prices_{freq}.csv")
            prices.to_csv(fname)
            print(f"  Saved to {fname}")

        return prices

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def fetch_fundamentals(
        self,
        tickers,
        period: str = "FY0",
        scale: int = 6,
        fields: dict | None = None,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch WorldScope annual fundamental data.

        Parameters
        ----------
        tickers  : list of RICs or constituent DataFrame
        period   : 'FY0' (most recent), 'FY-1', 'FY-2', etc.
        scale    : 0=units, 3=thousands, 6=millions (default)
        fields   : override default field dict {col_name: TR_field}
        output_dir : save CSV here if provided

        Returns
        -------
        DataFrame indexed by RIC.
        """
        ld = self._ld
        rics = self._ric_list(tickers)
        flds = fields or _FUNDAMENTAL_FIELDS

        tr_fields = list(flds.values())
        col_names = list(flds.keys())

        print(f"Fetching fundamentals for {len(rics):,} RICs (period={period})...")

        frames = []
        for batch in self._chunk(rics, 200):
            df = ld.get_data(
                universe=batch,
                fields=tr_fields,
                parameters={"Period": period, "Scale": scale, "SDate": 0},
            )
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=0)
        result.columns = ["ric"] + col_names
        result = result.set_index("ric")
        result = result.apply(pd.to_numeric, errors="coerce")

        print(f"  {result.shape[0]:,} RICs × {result.shape[1]:,} fields")
        print(f"  Coverage: {result.notna().mean().mean():.0%} non-null")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, "rl_fundamentals.csv")
            result.to_csv(fname)
            print(f"  Saved to {fname}")

        return result

    # ── Estimates ─────────────────────────────────────────────────────────────

    def fetch_estimates(
        self,
        tickers,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch IBES / StarMine consensus estimates (FY1 and FY2).

        Returns DataFrame indexed by RIC with columns for EPS, revenue,
        EBITDA estimates plus broker recommendation and target price.
        """
        ld = self._ld
        rics = self._ric_list(tickers)

        print(f"Fetching estimates for {len(rics):,} RICs...")

        # FY1 snapshot
        fy1_fields = {
            "eps_est_fy1":    "TR.EPSMeanEstimate",
            "eps_smart":      "TR.EPSSmartEstimate",
            "eps_surprise":   "TR.EPSSurprise",
            "eps_revision":   "TR.EPSRevisionsSurprise",
            "rev_est_fy1":    "TR.RevenueMeanEstimate",
            "rev_surprise":   "TR.RevenueSurprise",
            "ebitda_est_fy1": "TR.EBITDAMeanEstimate",
            "num_analysts":   "TR.NumberOfEstimates",
            "rec_mean":       "TR.TPMeanRec",
            "target_price":   "TR.TPMean",
        }

        fy2_fields = {
            "eps_est_fy2": "TR.EPSMeanEstimate",
            "rev_est_fy2": "TR.RevenueMeanEstimate",
        }

        frames_fy1, frames_fy2 = [], []
        for batch in self._chunk(rics, 200):
            df1 = ld.get_data(
                universe=batch,
                fields=list(fy1_fields.values()),
                parameters={"Period": "FY1", "SDate": 0},
            )
            df2 = ld.get_data(
                universe=batch,
                fields=["TR.EPSMeanEstimate", "TR.RevenueMeanEstimate"],
                parameters={"Period": "FY2", "SDate": 0},
            )
            frames_fy1.append(df1)
            frames_fy2.append(df2)

        fy1 = pd.concat(frames_fy1)
        fy1.columns = ["ric"] + list(fy1_fields.keys())
        fy1 = fy1.set_index("ric")

        fy2 = pd.concat(frames_fy2)
        fy2.columns = ["ric", "eps_est_fy2", "rev_est_fy2"]
        fy2 = fy2.set_index("ric")

        result = fy1.join(fy2, how="left")
        result = result.apply(pd.to_numeric, errors="coerce")

        # Implied PE on forward estimate
        fund = self.fetch_fundamentals(rics, fields={"market_cap": "TR.CompanyMarketCap"})
        if "market_cap" in fund.columns:
            shares = self.fetch_fundamentals(
                rics, fields={"shares": "TR.SharesOutstanding"}
            )["shares"]
            price = fund["market_cap"] / shares  # rough price proxy
            result["pe_forward"] = price / result["eps_est_fy1"]

        print(f"  {result.shape[0]:,} RICs × {result.shape[1]:,} fields")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, "rl_estimates.csv")
            result.to_csv(fname)
            print(f"  Saved to {fname}")

        return result

    # ── ESG ───────────────────────────────────────────────────────────────────

    def fetch_esg(
        self,
        tickers,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Refinitiv ESG scores (environment, social, governance + combined).

        Returns DataFrame indexed by RIC.
        """
        ld = self._ld
        rics = self._ric_list(tickers)

        print(f"Fetching ESG scores for {len(rics):,} RICs...")

        tr_fields = list(_ESG_FIELDS.values())
        col_names = list(_ESG_FIELDS.keys())

        frames = []
        for batch in self._chunk(rics, 200):
            df = ld.get_data(universe=batch, fields=tr_fields)
            frames.append(df)

        result = pd.concat(frames)
        result.columns = ["ric"] + col_names
        result = result.set_index("ric")

        # Numeric columns only (esg_year stays as date string)
        num_cols = [c for c in result.columns if c != "esg_year"]
        result[num_cols] = result[num_cols].apply(pd.to_numeric, errors="coerce")

        coverage = result["esg_score"].notna().mean()
        print(f"  ESG coverage: {coverage:.0%} of universe")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, "rl_esg.csv")
            result.to_csv(fname)
            print(f"  Saved to {fname}")

        return result

    # ── Convenience: fetch all ────────────────────────────────────────────────

    def fetch_all(
        self,
        tickers,
        output_dir: str | None = None,
        prices: bool = True,
        fundamentals: bool = True,
        estimates: bool = True,
        esg: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all data types in one call. Returns dict with keys:
        'prices', 'fundamentals', 'estimates', 'esg'.
        """
        result = {}
        rics = self._ric_list(tickers)

        if prices:
            result["prices"] = self.fetch_prices(rics, output_dir=output_dir)
        if fundamentals:
            result["fundamentals"] = self.fetch_fundamentals(rics, output_dir=output_dir)
        if estimates:
            result["estimates"] = self.fetch_estimates(rics, output_dir=output_dir)
        if esg:
            result["esg"] = self.fetch_esg(rics, output_dir=output_dir)

        return result


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_RICS = ["AAPL.O", "MSFT.O", "BP.L", "ASML.AS", "7203.T"]

    with RefinitivClient() as rl:
        data = rl.fetch_all(TEST_RICS, output_dir="data")

    for k, df in data.items():
        print(f"\n{k}: {df.shape}")
        print(df.head(3))
