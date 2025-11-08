# main/management/commands/fetch_weather.py
from django.core.management.base import BaseCommand
from django.core.cache import cache
import logging
from config.settings import WEATHER_CACHE_KEY, WEATHER_CACHE_TTL
from main.utils.weather import fetch_owm_once

LOG = logging.getLogger(__name__)

CACHE_KEY = WEATHER_CACHE_KEY
CACHE_TTL = int(WEATHER_CACHE_TTL)

class Command(BaseCommand):
    help = "Fetch current outside temperature and store in Django cache (and optional JSON file)."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Run once and exit")

    def handle(self, *args, **options):
        try:
            payload = fetch_owm_once()
            cache.set(CACHE_KEY, {"ok": True, "data": payload}, timeout=CACHE_TTL)
            self.stdout.write(self.style.SUCCESS(f"Fetched temp: {payload.get('temp')} {payload.get('units')}"))
        except Exception as e:
            LOG.exception("Weather fetch failed")
            cache.set(CACHE_KEY, {"ok": False, "error": str(e)}, timeout=CACHE_TTL)
            self.stderr.write(self.style.ERROR(f"Fetch failed: {e}"))
