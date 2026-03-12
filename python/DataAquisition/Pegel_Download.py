import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
DOMAIN       = "https://www.pegelonline.wsv.de"
STATION_URL  = f"{DOMAIN}/webservices/files/Wasserstand+Rohdaten/ELBE/HAMBURG+ST.+PAULI"
UUID_URL     = f"{DOMAIN}/webservices/files/Wasserstand+Rohdaten/ELBE/d488c5cc-4de9-4631-8ce1-0db0e700b546"
OUTPUT_CSV   = "hamburg_stpauli_waterlevel_2018_2025.csv"
TZ           = "Etc/GMT-1"                           # CET all-year, no DST

# ─── Step 1: Scrape ALL links from station directory page ─────────────────────
def list_station_entries(base_url: str) -> list[str]:
    resp = requests.get(base_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    base_path = base_url.replace(DOMAIN, "").rstrip("/") + "/"
    entries = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Normalise to absolute path on domain
        if href.startswith(base_path):
            folder = href[len(base_path):].split("/")[0].strip()
            if folder:
                entries.append((folder, href))
    return entries

# ─── Step 2: Identify historical vs. recent entries ───────────────────────────
# Recent daily files: folder matches "DD.MM.YYYY"
# Historical archives: folder is a Java timestamp  e.g. "Mon Jan 06 09:00:00 CET 2026"
DATE_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")

def categorise(entries: list[tuple]) -> tuple[list, list]:
    recent, historical = [], []
    for folder, href in entries:
        (recent if DATE_PATTERN.match(folder) else historical).append((folder, href))
    return recent, historical

# ─── Step 3: Parse ZRXP / plain-text pegelonline format ───────────────────────
#  Line format: YYYYMMDDHHMMSS VALUE   (CET = UTC+1, no DST)
def parse_raw_text(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        # "YYYYMMDDHHMMSS VALUE" or "YYYYMMDDHHMMSS;VALUE;..."
        m = re.match(r"^(\d{14})\s+([\d.\-]+)$", line)
        if not m:
            parts = line.split(";")
            if len(parts) >= 2 and re.match(r"^\d{14}$", parts[0].strip()):
                m_ts, m_val = parts[0].strip(), parts[1].strip()
            else:
                continue
        else:
            m_ts, m_val = m.group(1), m.group(2)
        try:
            v = float(m_val)
            if v in (-999, -9999):
                continue
            records.append({"datetime": pd.to_datetime(m_ts, format="%Y%m%d%H%M%S"),
                             "water_level_cm": v})
        except ValueError:
            pass
    return records

# ─── Step 4: Download a single file ────────────────────────────────────────────
def download_and_parse(url: str) -> list[dict]:
    try:
        log.info(f"  GET {url}")
        r = requests.get(url, timeout=300)
        if r.status_code == 200 and len(r.content) > 200:
            text = r.content.decode("latin-1", errors="replace")
            recs = parse_raw_text(text)
            log.info(f"  → {len(recs):,} raw records")
            return recs
        log.warning(f"  → HTTP {r.status_code}, {len(r.content)} bytes — skipping")
    except requests.RequestException as e:
        log.warning(f"  → Request failed: {e}")
    return []

# ─── Main ─────────────────────────────────────────────────────────────────────
log.info("=== PegelOnline Historical Archive – Hamburg St. Pauli 2018–2025 ===")

all_records = []

for label, base_url in [("name-based", STATION_URL), ("uuid-based", UUID_URL)]:
    log.info(f"\n--- Directory listing ({label}) ---")
    try:
        entries = list_station_entries(base_url)
    except Exception as e:
        log.warning(f"  Listing failed: {e}")
        continue

    recent, historical = categorise(entries)
    log.info(f"  Total sub-entries : {len(entries)}")
    log.info(f"  Recent daily files: {len(recent)}")
    log.info(f"  Historical archive: {len(historical)}  → {[f for f,_ in historical]}")

    for folder, href in historical:
        # The historical file sits at {folder}/down.txt  (same as daily files)
        url = DOMAIN + href.rstrip("/") + "/down.txt"
        recs = download_and_parse(url)
        all_records.extend(recs)

    if all_records:
        log.info(f"  ✓ Got data from {label} listing, stopping.")
        break

# ─── Filter, build DataFrame, save ────────────────────────────────────────────
if not all_records:
    log.error("\nNo historical archive found in directory. Possible reasons:")
    log.error("  • WSV has not yet generated the historical file for this station")
    log.error("  • Station archive requires authenticated 'Abo' subscription")
    log.error("\nNext best option — request data directly from BfG:")
    log.error("  Email: Datenstelle-M1@bafg.de  (subject: Pegeldaten Hamburg St. Pauli 2018–2025)")
    log.error("  Web  : https://geoportal.bafg.de/OpenData/")
else:
    df = pd.DataFrame(all_records)
    df["datetime"] = df["datetime"].dt.tz_localize(TZ)
    mask = (df["datetime"] >= pd.Timestamp("2018-01-01", tz=TZ)) & \
           (df["datetime"] <= pd.Timestamp("2026-01-01", tz=TZ) - pd.Timedelta(seconds=1))
    df = (df[mask]
          .sort_values("datetime")
          .drop_duplicates(subset="datetime")
          .reset_index(drop=True))
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"\n✓  Saved {len(df):,} records → {OUTPUT_CSV}")
    log.info(f"   Range   : {df['datetime'].min()} → {df['datetime'].max()}")
    log.info(f"   Sample  :\n{df.head(5).to_string()}")
