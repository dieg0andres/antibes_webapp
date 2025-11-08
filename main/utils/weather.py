# main/utils/weather.py
import os
import sys
from pathlib import Path
import requests
from datetime import datetime, timezone

# Ensure project root is on sys.path when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# At minimum set DJANGO_SETTINGS_MODULE so config imports succeed
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from config.settings import OWM_API_KEY

# Defaults; override via env vars
ZIP = "77082,us"
UNITS = "imperial"  # "imperial" => Fahrenheit
DEFAULT_TIMEOUT = 10
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_owm_once(zip_code=ZIP, units=UNITS, timeout=DEFAULT_TIMEOUT):
    """Fetch weather from OpenWeatherMap and return normalized dict."""
    if not OWM_API_KEY:
        raise RuntimeError("OWM_API_KEY not set in environment")
    params = {"zip": zip_code, "units": units, "appid": OWM_API_KEY}
    headers = {"User-Agent": "antibes_webapp/raspi (diego.a.galindo@gmail.com)"}
    r = requests.get(OWM_URL, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    j = r.json()
    temp = j.get("main", {}).get("temp")
    return {
        "temp": temp,
        "units": "F" if units == "imperial" else "C",
        "station": j.get("name"),
        "tz_offset_sec": j.get("timezone"),
        "raw": j,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }