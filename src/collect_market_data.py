from __future__ import annotations

from io import StringIO
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

from .config import END_DATE, INTERIM, PROCESSED, START_DATE, ensure_directories
from .utils import save_csv

FRED_SERIES = ["T10Y2Y", "DGS1", "DGS3MO"]
YAHOO_TICKERS = {"DX-Y.NYB": "dxy", "IWF": "iwf", "IWN": "iwn"}


def fred_series(series_id: str, start: str, end: str) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    response = requests.get(
        url,
        params={"id": series_id, "cosd": start, "coed": end},
        timeout=60,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), na_values=".")
    frame.columns = ["date", series_id.lower()]
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")[series_id.lower()].astype(float)


def yahoo_chart_series(ticker: str, start: str, end: str) -> pd.Series:
    """Crumb-free fallback for environments where yfinance is rate-limited."""
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
    response = requests.get(
        url,
        params={"period1": period1, "period2": period2, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    quote_data = result["indicators"].get("adjclose", result["indicators"]["quote"])[0]
    values = quote_data.get("adjclose", quote_data.get("close"))
    return pd.Series(values, index=index, name=YAHOO_TICKERS[ticker], dtype=float)


def collect(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    ensure_directories()
    padded_start = str((pd.Timestamp(start) - pd.Timedelta(days=10)).date())
    padded_end = str((pd.Timestamp(end) + pd.Timedelta(days=7)).date())
    frame = pd.concat(
        [fred_series(series, padded_start, padded_end) for series in FRED_SERIES], axis=1
    )
    prices = yf.download(
        list(YAHOO_TICKERS), start=padded_start, end=str(pd.Timestamp(padded_end) + pd.Timedelta(days=1)),
        auto_adjust=True, progress=False, group_by="column",
    )["Close"]
    if prices.empty or set(YAHOO_TICKERS) - set(prices.columns):
        print("Bulk Yahoo download incomplete; using the chart endpoint ticker by ticker.")
        prices = pd.concat(
            [yahoo_chart_series(ticker, padded_start, padded_end) for ticker in YAHOO_TICKERS], axis=1
        )
    else:
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.rename(columns=YAHOO_TICKERS)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
    frame = frame.join(prices, how="outer").sort_index()
    frame.index.name = "date"
    frame = frame.reset_index()
    save_csv(frame, INTERIM / "market_daily.csv")
    return frame


def _event_day(trading_dates: pd.DatetimeIndex, release: pd.Timestamp) -> pd.Timestamp:
    date = release.tz_convert("America/New_York").tz_localize(None).normalize()
    same_day_is_trading = date in trading_dates
    before_close = release.tz_convert("America/New_York").hour < 16
    candidates = trading_dates[trading_dates >= date]
    if same_day_is_trading and before_close:
        return date
    later = trading_dates[trading_dates > date]
    if later.empty:
        raise ValueError(f"No market observation after {release}")
    return later[0]


def construct_events() -> pd.DataFrame:
    scored_path = INTERIM / "documents_scored.csv"
    if not scored_path.exists():
        raise FileNotFoundError("Run `python -m src.score_finbert` before constructing events.")
    docs = pd.read_csv(scored_path)
    market = pd.read_csv(INTERIM / "market_daily.csv", parse_dates=["date"]).set_index("date")
    trading = pd.DatetimeIndex(market[["dxy", "iwf", "iwn"]].dropna(how="all").index).sort_values()
    rows = []
    for doc in docs.itertuples(index=False):
        release = pd.Timestamp(doc.release_datetime_et)
        day = _event_day(trading, release)
        previous_candidates = trading[trading < day]
        if previous_candidates.empty:
            continue
        previous = previous_candidates[-1]
        current_values = market.loc[:day].ffill().loc[day]
        prior_values = market.loc[:previous].ffill().loc[previous]
        rows.append(
            {
                "document_id": doc.document_id,
                "event_day": day,
                "prior_trading_day": previous,
                "dxy_change_pct": 100 * (current_values.dxy / prior_values.dxy - 1),
                "spread_10s2s_change_bp": 100 * (current_values.t10y2y - prior_values.t10y2y),
                "dgs1_change_bp": 100 * (current_values.dgs1 - prior_values.dgs1),
                "growth_minus_value_pct": 100
                * ((current_values.iwf / prior_values.iwf - 1) - (current_values.iwn / prior_values.iwn - 1)),
                "dgs3mo_change_bp": 100 * (current_values.dgs3mo - prior_values.dgs3mo),
            }
        )
    events = docs.merge(pd.DataFrame(rows), on="document_id", how="left", validate="one_to_one")
    save_csv(events, PROCESSED / "event_sample.csv")
    return events


if __name__ == "__main__":
    collect()
    construct_events()
