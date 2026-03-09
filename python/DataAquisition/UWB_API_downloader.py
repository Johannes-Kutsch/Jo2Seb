#!/usr/bin/env python3
"""
Download air quality data from Umweltbundesamt Luftdaten API v4
Component: 1 (PM10), Scope: 6 (daily mean / Tagesmittelwert)
Stations: list below | Timespan: 2018-2025

Response structure (from API indices field):
  data[station_id][date_start] = [component_id, scope_id, value, date_end, index]
"""

import requests
import sqlite3
import time
import logging
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from os import path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Available components: ─────────────────────────────────────────────────────
''' Other components are:
    {'1': ['1', 'PM10', 'PM₁₀', 'µg/m³', 'Feinstaub'],
    '2': ['2', 'CO', 'CO', 'mg/m³', 'Kohlenmonoxid'],
    '3': ['3', 'O3', 'O₃', 'µg/m³', 'Ozon'],
    '4': ['4', 'SO2', 'SO₂', 'µg/m³', 'Schwefeldioxid'],
    '5': ['5', 'NO2', 'NO₂', 'µg/m³', 'Stickstoffdioxid'],
    '6': ['6', 'PM10PB', 'Pb', 'µg/m³', 'Blei im Feinstaub'],
    '7': ['7', 'PM10BAP', 'BaP', 'ng/m³', 'Benzo(a)pyren im Feinstaub'],
    '8': ['8', 'CHB', 'C₆H₆', 'µg/m³', 'Benzol'],
    '9': ['9', 'PM2', 'PM₂,₅', 'µg/m³', 'Feinstaub'],
    '10': ['10', 'PM10AS', 'As', 'ng/m³', 'Arsen im Feinstaub'],
    '11': ['11', 'PM10CD', 'Cd', 'ng/m³', 'Cadmium im Feinstaub'],
    '12': ['12', 'PM10NI', 'Ni', 'ng/m³', 'Nickel im Feinstaub']}
    '''

# ── Available scopes: ─────────────────────────────────────────────────────

'''
    1:  Tagesmittel
    2:  Ein-Stunden-Mittelwert
    3:  Ein-Stunden-Tagesmaxima
    4:  Acht-Stunden-Mittelwert
    5:  Acht-Stunden-Tagesmaxima
    6:  stündlich gleitendes Tagesmittel

'''

# ── Available combinations of components and Scopes: ──────────────────────────
'''
component      |     scope
--------------------------
1               | 1, 6
2               | 4, 5
3               | 2, 3, 4, 5
4               | 1, 2, 3
5               | 2, 3, 
'''

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL      = "https://luftdaten.umweltbundesamt.de/api/air-data/v4/measures/json"
STATIONS      = ['784', '791', '802', '835', '840', '844', '846',
                 '855', '857', '10348', '826', '10466']
#STATIONS      = ['784']
COMPONENT     = 6
SCOPE         = 6
DATE_FROM     = date(2018, 1, 1)
DATE_TO       = date(2025, 12, 31)
CHUNK_MONTHS  = 12
RETRY_MAX     = 3
RETRY_WAIT    = 5
REQUEST_DELAY = 0.5

db_folder = path.join(".", 'files/data/lfs/UWB')

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db_connection(station_id: str, component: int, db_folder: str, scope: int ) -> sqlite3.Connection:
    db_name = f"station_{station_id}_comp_{component}_scope_{scope}.db"
    db_path = path.join(db_folder, db_name)
    print(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            date_start   TEXT,
            date_end     TEXT,
            station_id   INTEGER,
            component_id INTEGER,
            scope_id     INTEGER,
            value        REAL,
            index_value  TEXT,
            PRIMARY KEY (date_start, station_id, component_id, scope_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON measurements (date_start)")
    conn.commit()
    return conn


def insert_records(conn: sqlite3.Connection, records: list):
    conn.executemany(
        """INSERT OR REPLACE INTO measurements
           (date_start, date_end, station_id, component_id, scope_id, value, index_value)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        records,
    )
    conn.commit()


# ── Parse response ────────────────────────────────────────────────────────────
def parse_data(payload: dict) -> list:
    """
    Actual API structure (confirmed from live response + indices field):

      data[station_id][date_start] = [component_id, scope_id, value, date_end, index]
                                       row[0]        row[1]   row[2]  row[3]   row[4]
    """
    records = []
    data_block = payload.get("data", {})

    for station_id, date_dict in data_block.items():
        if not isinstance(date_dict, dict):
            continue
        for date_start, row in date_dict.items():
            if not isinstance(row, list) or len(row) < 4:
                continue
            component_id = row[0]
            scope_id     = row[1]
            value        = row[2]
            date_end     = row[3]
            index_value  = row[4] if len(row) > 4 else None

            records.append((
                date_start,
                date_end,
                station_id,
                component_id,
                scope_id,
                value,
                index_value,
            ))
    return records


# ── API fetch ─────────────────────────────────────────────────────────────────
def fetch_chunk(station: str, date_from: date, date_to: date) -> list:
    params = {
        "date_from":  date_from.strftime("%Y-%m-%d"),
        "date_to":    date_to.strftime("%Y-%m-%d"),
        "time_from":  1,
        "time_to":    24,
        "station":    station,
        "component":  COMPONENT,
        "scope":      SCOPE,
    }

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            logger.warning(
                "Station %s | %s→%s | attempt %d/%d failed: %s",
                station, date_from, date_to, attempt, RETRY_MAX, exc,
            )
            if attempt == RETRY_MAX:
                logger.error("Giving up on station %s chunk %s→%s",
                             station, date_from, date_to)
                return []
            time.sleep(RETRY_WAIT)

    return parse_data(payload)


# ── Date chunk generator ──────────────────────────────────────────────────────
def generate_chunks(start: date, end: date, months: int):
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(
            chunk_start + relativedelta(months=months) - timedelta(days=1),
            end,
        )
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for station in STATIONS:
        logger.info("=== Processing station %s ===", station)
        conn = get_db_connection(station, COMPONENT, db_folder=db_folder, scope=SCOPE)
        total_records = 0

        for chunk_from, chunk_to in generate_chunks(DATE_FROM, DATE_TO, CHUNK_MONTHS):
            logger.info("Station %s | fetching %s → %s",
                        station, chunk_from, chunk_to)
            records = fetch_chunk(station, chunk_from, chunk_to)
            if records:
                insert_records(conn, records)
                total_records += len(records)
                logger.info("  → %d records inserted", len(records))
            else:
                logger.info("  → no data returned")
            time.sleep(REQUEST_DELAY)

        conn.close()
        logger.info(
            "Station %s done. Total records: %d  →  station_%s_component_%d.db",
            station, total_records, station, COMPONENT,
        )

    logger.info("All stations completed.")


if __name__ == "__main__":
    main()
