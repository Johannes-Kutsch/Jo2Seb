from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
GFW = PROJECT_ROOT / "files/data/lfs/GFW"
DWD = PROJECT_ROOT / "files/data/lfs/DWD"
VOYAGES_DIR = GFW / "voyages"
WEATHER_DIR = DWD / "weather"