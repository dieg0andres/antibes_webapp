from datetime import date
from typing import Dict, Optional
from django.core.cache import cache
from config.settings import WEATHER_CACHE_KEY, WEATHER_CACHE_TTL
MORTGAGE_END_DATE = date(2027, 12, 1)

MORTGAGE_SCHEDULE: Dict[str, Dict[str, int]] = {
    'Jul 2025': {'principal': 400000, 'interest_paid': 0},
    'Aug 2025': {'principal': 399641, 'interest_paid': 2250},
    'Sep 2025': {'principal': 399280, 'interest_paid': 4498},
    'Oct 2025': {'principal': 398028, 'interest_paid': 6744},
    'Nov 2025': {'principal': 390490, 'interest_paid': 8983},
    'Dec 2025': {'principal': 380908, 'interest_paid': 11179},
    'Jan 2026': {'principal': 380162, 'interest_paid': 13322},
    'Feb 2026': {'principal': 364412, 'interest_paid': 15460},
    'Mar 2026': {'principal': 348573, 'interest_paid': 17510},
    'Apr 2026': {'principal': 332644, 'interest_paid': 19471},
    'May 2026': {'principal': 316627, 'interest_paid': 21342},
    'Jun 2026': {'principal': 300519, 'interest_paid': 23123},
    'Jul 2026': {'principal': 284320, 'interest_paid': 24813},
    'Aug 2026': {'principal': 268031, 'interest_paid': 26413},
    'Sep 2026': {'principal': 251649, 'interest_paid': 27920},
    'Oct 2026': {'principal': 235176, 'interest_paid': 29336},
    'Nov 2026': {'principal': 218610, 'interest_paid': 30659},
    'Dec 2026': {'principal': 201951, 'interest_paid': 31889},
    'Jan 2027': {'principal': 185198, 'interest_paid': 33024},
    'Feb 2027': {'principal': 168351, 'interest_paid': 34066},
    'Mar 2027': {'principal': 151409, 'interest_paid': 35013},
    'Apr 2027': {'principal': 134372, 'interest_paid': 35865},
    'May 2027': {'principal': 117239, 'interest_paid': 36621},
    'Jun 2027': {'principal': 100009, 'interest_paid': 37280},
    'Jul 2027': {'principal': 82683, 'interest_paid': 37843},
    'Aug 2027': {'principal': 65259, 'interest_paid': 38308},
    'Sep 2027': {'principal': 47737, 'interest_paid': 38675},
    'Oct 2027': {'principal': 30117, 'interest_paid': 38943},
    'Nov 2027': {'principal': 12398, 'interest_paid': 39113},
    'Dec 2027': {'principal': 0, 'interest_paid': 39183},
}

_LAST_SNAPSHOT = list(MORTGAGE_SCHEDULE.values())[-1]
DEFAULT_SNAPSHOT = {
    'principal': 0,
    'interest_paid': _LAST_SNAPSHOT.get('interest_paid', 0),
}


def _build_month_key(target_date: date) -> str:
    return target_date.strftime('%b %Y')


def _resolve_snapshot_for_date(target_date: date) -> Dict[str, int]:
    month_key = _build_month_key(target_date)
    snapshot = MORTGAGE_SCHEDULE.get(month_key)
    if snapshot is not None:
        return dict(snapshot)

    return dict(DEFAULT_SNAPSHOT)


def _get_outside_temp_from_cache():
    cache_data = cache.get(WEATHER_CACHE_KEY)
    if cache_data is not None:
        return cache_data.get('data').get('temp')
    return "Error getting outside temperature from cache"

def build_mortgage_dashboard(today: Optional[date] = None) -> Dict[str, int]:
    current_date = today or date.today()
    snapshot = _resolve_snapshot_for_date(current_date)

    days_finish_mortgage = (MORTGAGE_END_DATE - current_date).days
    outside_temp = _get_outside_temp_from_cache()

    return {
        'days_finish_mortgage': days_finish_mortgage,
        'principal': snapshot['principal'],
        'interest_paid': snapshot['interest_paid'],
        'outside_temp': int(outside_temp),
    }

