from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
GFW = PROJECT_ROOT / "files/data/lfs/GFW"
DWD = PROJECT_ROOT / "files/data/lfs/DWD"
UWB = PROJECT_ROOT / "files/data/lfs/UWB"
PEGEL = PROJECT_ROOT / "files/data/lfs/Pegel"
COPERNICUS = PROJECT_ROOT / "files/data/lfs/Copernicus"
UWB_DATA = UWB / "data_tables"
UWB_METADATA = UWB / "metadata"
VOYAGES_DIR = GFW / "voyages"
WEATHER_DIR = DWD / "weather"
