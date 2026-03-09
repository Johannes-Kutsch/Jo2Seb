"""
Hamburg DWD Hourly Weather Downloader
--------------------------------------
Data source: Bright Sky API (api.brightsky.dev) — the official free JSON API
for DWD open weather data (same data as dwd.api.bund.dev / opendata.dwd.de).

NOTE: dwd.api.bund.dev only exposes the DWD App's stationOverviewExtended
endpoint, which provides current/near-forecast data only — no historical
archive back to 2018. Bright Sky provides full historical DWD observations.

Requirements:
    pip install requests pandas tqdm

Output:
    ./hamburg_weather_data/<StationName>_<StationID>.csv
    Each CSV has a pandas DatetimeIndex (hourly, Europe/Berlin timezone)
    and one column per weather variable.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from os import path
import requests
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

#--------Directories ---------------------------------------
## Project directory
project_dir = path.dirname(path.dirname(path.abspath('')))
## Location for data_tables
data_tables_dir = path.join(project_dir, "files", "data", "lfs", "DWD", "data_tables")

BASE_URL       = "https://api.brightsky.dev"
OUTPUT_DIR     = data_tables_dir

# Hamburg city centre
HAMBURG_LAT    = 53.5753
HAMBURG_LON    = 10.0153
HAMBURG_RADIUS = 25_000          # metres (~25 km covers all HH districts)

START_DATE     = datetime(2018, 1, 1, tzinfo=timezone.utc)
END_DATE       = datetime.now(timezone.utc)

# Columns to drop (metadata / redundant)
DROP_COLS = {"source_id", "icon", "condition", "fallback_source_ids"}

REQUEST_DELAY  = 0.4             # seconds between API calls (be polite)
MAX_RETRIES    = 4
RETRY_BACKOFF  = 2               # exponential back-off base in seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> dict:
    """GET with automatic retry / exponential back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return {}            # no data for this period — not an error
            wait = RETRY_BACKOFF ** attempt
            log.warning("HTTP %s on attempt %d/%d — retrying in %ds",
                        exc.response.status_code if exc.response else "?",
                        attempt, MAX_RETRIES, wait)
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF ** attempt
            log.warning("Request error %s on attempt %d/%d — retrying in %ds",
                        exc, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    log.error("All retries exhausted for %s %s", url, params)
    return {}


def month_ranges(start: datetime, end: datetime):
    """Yield (month_start, month_end) pairs covering [start, end)."""
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        yield max(current, start), min(next_month, end)
        current = next_month


# ── Station discovery ─────────────────────────────────────────────────────────

def get_hamburg_stations() -> list[dict]:
    """
    Return unique DWD stations within HAMBURG_RADIUS of Hamburg centre.
    Queries Bright Sky /sources for historical + recent + current sources.
    """
    log.info("Querying Bright Sky /sources for Hamburg stations …")
    data = _get(f"{BASE_URL}/sources", {
        "lat":    HAMBURG_LAT,
        "lon":    HAMBURG_LON,
        "radius": HAMBURG_RADIUS,
    })
    sources = data.get("sources", [])

    # Keep one entry per DWD station ID (prefer 'historical' observation type)
    priority = {"historical": 0, "recent": 1, "current": 2, "forecast": 3}
    best: dict[str, dict] = {}
    for src in sources:
        sid = src.get("dwd_station_id")
        if not sid:
            continue
        obs = src.get("observation_type", "forecast")
        if sid not in best or priority.get(obs, 9) < priority.get(
                best[sid].get("observation_type", "forecast"), 9):
            best[sid] = src

    stations = list(best.values())
    log.info("Found %d unique DWD stations near Hamburg", len(stations))
    for s in stations:
        log.info("  %-30s  ID=%s  type=%-12s  dist=%.0f m",
                 s.get("station_name", "?"),
                 s["dwd_station_id"],
                 s.get("observation_type", "?"),
                 s.get("distance", 0))
    return stations


# ── Data download ─────────────────────────────────────────────────────────────

def fetch_station_data(dwd_station_id: str) -> pd.DataFrame:
    """
    Download all hourly records for one station between START_DATE and END_DATE.
    Requests are chunked month-by-month to stay within API limits.
    Returns a DataFrame with a DatetimeIndex (Europe/Berlin, hourly).
    """
    ranges = list(month_ranges(START_DATE, END_DATE))
    all_records: list[dict] = []

    for month_start, month_end in tqdm(ranges,
                                       desc=f"  {dwd_station_id}",
                                       unit="month",
                                       leave=False):
        data = _get(f"{BASE_URL}/weather", {
            "dwd_station_id": dwd_station_id,
            "date":      month_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_date": month_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "units":     "dwd",          # SI / DWD native units
        })
        records = data.get("weather", [])
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # ── Index: UTC → Europe/Berlin, snapped to full hour ──────────────────
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
          .dt.tz_convert("Europe/Berlin")
    )
    df = df.set_index("timestamp")
    df.index = df.index.floor("h", ambiguous='infer', nonexistent='shift_forward')
    df.index.name = "datetime"

    # ── Drop metadata columns ──────────────────────────────────────────────
    df.drop(columns=[c for c in DROP_COLS if c in df.columns],
            inplace=True, errors="ignore")

    # ── Sort and de-duplicate (keep first in case of overlapping sources) ──
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stations = get_hamburg_stations()
    if not stations:
        log.error("No stations found — check coordinates / API availability.")
        return

    summary_rows = []

    for station in stations:
        sid   = station["dwd_station_id"]
        sname = (station.get("station_name") or f"station_{sid}") \
                    .replace("/", "-").replace(" ", "_")
        fname = os.path.join(OUTPUT_DIR, f"{sname}_{sid}.csv")

        # ── Skip already-downloaded files ──────────────────────────────────
        if os.path.exists(fname):
            log.info("SKIP  %s — file already exists", fname)
            existing = pd.read_csv(fname, index_col=0, parse_dates=True)
            summary_rows.append({"station_id": sid,
                                  "station_name": sname,
                                  "records": len(existing),
                                  "file": fname,
                                  "status": "skipped"})
            continue

        log.info("Downloading  %-30s  (ID %s) …", sname, sid)
        df = fetch_station_data(sid)

        if df.empty:
            log.warning("  → No data returned for station %s", sid)
            summary_rows.append({"station_id": sid,
                                  "station_name": sname,
                                  "records": 0,
                                  "file": "",
                                  "status": "no_data"})
            continue

        df.to_csv(fname)
        log.info("  → Saved %d hourly records  →  %s", len(df), fname)
        summary_rows.append({"station_id": sid,
                              "station_name": sname,
                              "records": len(df),
                              "file": fname,
                              "status": "ok"})

    # ── Write summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "_download_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    log.info("\nDone. Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
