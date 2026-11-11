"""Market data for the empirical application (paper Sec. 6).

OHLCV bars come only from the QuantX Dataloader REST API. Treasury yields and
CPI inflation (used to build the macro-financial context, a documented
substitute for Goyal et al. 2024, see results/RESULTS.md) come from the
Massive REST API, whose base URL and key are described below. Downloaded data
is cached under ``data/cache/`` (ignored by git) and never committed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_API_ROOT = "https://api.massive.com"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataLoaderError(RuntimeError):
    """A failed Dataloader or Massive request, or a load with no data at all."""


def _dataloader_headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataLoaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _massive_headers() -> dict[str, str]:
    load_dotenv()
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not massive_key:
        raise DataLoaderError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {massive_key}"}


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _dataloader_headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, DATALOADER_API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataLoaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataLoaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Bars for ``tickers`` over the inclusive window, and load metadata.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus metadata with every ticker's
    ``ticker_details`` entry. The load fails only when no ticker loads.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"ohlcv_{key}.csv.gz", CACHE_DIR / f"ohlcv_{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataLoaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def _massive_get(route: str, params: dict) -> list[dict]:
    """Paginate a Massive REST endpoint and return the concatenated ``results``."""
    headers = _massive_headers()
    url, rows = MASSIVE_API_ROOT + route, []
    seen_urls = set()
    while url and url not in seen_urls:
        seen_urls.add(url)
        response = requests.get(url, params=params if url == MASSIVE_API_ROOT + route else None,
                                 headers=headers, timeout=60)
        if response.status_code != 200:
            raise DataLoaderError(f"{route}: HTTP {response.status_code}: {response.text[:200]}")
        payload = response.json()
        rows.extend(payload.get("results") or [])
        url = payload.get("next_url")
    return rows


def load_treasury_yields(start_date: str, end_date: str, *, use_cache: bool = True) -> pd.DataFrame:
    """Daily U.S. Treasury yields (Massive ``/fed/v1/treasury-yields``).

    Used to build the term spread (10-year minus 3-month yield) that
    substitutes for the term-spread component of the Goyal et al. (2024)
    context in the macro-financial context (see results/RESULTS.md).
    """
    key = hashlib.sha256(f"treasury_{start_date}_{end_date}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    rows = _massive_get("/fed/v1/treasury-yields", {
        "date.gte": start_date, "date.lte": end_date, "limit": 50000,
        "sort": "date.asc",
    })
    if not rows:
        raise DataLoaderError("treasury-yields: empty result for the requested window")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_inflation(start_date: str, end_date: str, *, use_cache: bool = True) -> pd.DataFrame:
    """Monthly U.S. CPI inflation (Massive ``/fed/v1/inflation``).

    Substitutes for the inflation component of the Goyal et al. (2024)
    context (see results/RESULTS.md).
    """
    key = hashlib.sha256(f"inflation_{start_date}_{end_date}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    rows = _massive_get("/fed/v1/inflation", {
        "date.gte": start_date, "date.lte": end_date, "limit": 50000,
        "sort": "date.asc",
    })
    if not rows:
        raise DataLoaderError("inflation: empty result for the requested window")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# Candidate pool of large, liquid U.S. common stocks used as a stand-in for the
# historical S&P 500 membership list, which is distributed through CRSP/WRDS
# and unavailable through the Dataloader or Massive APIs (see
# results/RESULTS.md, "Deviations from the paper"). The evaluation universe is
# cut from this pool by trailing dollar volume (paper Sec. 6.1 selection rule
# substitute), never by list order.
SP500_CANDIDATE_POOL = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "BRK.B", "UNH",
    "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT",
    "DHR", "LIN", "NEE", "TXN", "PM", "ORCL", "VZ", "ADBE", "CRM", "NKE",
    "WFC", "BMY", "RTX", "UPS", "QCOM", "HON", "T", "LOW", "INTC", "COP",
    "AMD", "CAT", "SPGI", "INTU", "AMGN", "DE", "BA", "GE", "IBM", "AXP",
    "PFE", "GS", "BKNG", "MDT", "ELV", "SBUX", "BLK", "GILD", "ISRG", "ADI",
    "LMT", "SYK", "MDLZ", "C", "TJX", "MMC", "SCHW", "CVS", "ADP", "VRTX",
    "CI", "ZTS", "MO", "PLD", "CB", "SO", "DUK", "BDX", "CL", "ITW",
    "EOG", "SLB", "APD", "NOC", "MMM", "FDX", "GD", "EMR", "USB", "PNC",
    "TGT", "ETN", "AON", "MCK", "KLAC", "SHW", "ICE", "FISV", "CSX", "D",
    "NSC", "HUM", "MPC", "PSX", "VLO", "OXY", "KMB", "GM", "F", "DOW",
    "APH", "ROP", "ECL", "MSI", "PGR", "TRV", "AIG", "ALL", "MET", "AFL",
    "PRU", "HCA", "REGN", "BIIB", "ADSK", "CDNS", "SNPS", "MCHP", "LRCX", "ORLY",
    "AZO", "ROST", "YUM", "DG", "DLTR", "KR", "SYY", "GIS", "K", "HSY",
]


def load_sp500_universe(top_n: int, start_date: str, end_date: str,
                         lookback_months: int = 12, *, use_cache: bool = True) -> pd.DataFrame:
    """Rank ``SP500_CANDIDATE_POOL`` by trailing dollar volume and keep ``top_n``.

    Returns one row per kept ticker with its average daily dollar volume over
    the trailing ``lookback_months`` ending at ``start_date``.
    """
    rank_start = (pd.Timestamp(start_date) - pd.DateOffset(months=lookback_months)).strftime("%Y-%m-%d")
    rank_end = (pd.Timestamp(start_date) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    bars, _ = load_ohlcv(SP500_CANDIDATE_POOL, rank_start, rank_end, "1d", use_cache=use_cache)
    bars["dollar_volume"] = bars["close"] * bars["volume"]
    ranked = (bars.groupby("ticker")["dollar_volume"].mean()
              .sort_values(ascending=False).reset_index())
    return ranked.head(top_n)
