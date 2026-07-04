from datetime import date
from typing import Dict, Optional
from django.core.cache import cache
from config.settings import WEATHER_CACHE_KEY, MOTIVATION_CACHE_KEY


MORTGAGE_END_DATE = date(2027, 2, 1)

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
    'Apr 2026': {'principal': 243755, 'interest_paid': 19471},
    'May 2026': {'principal': 233349, 'interest_paid': 20842},
    'Jun 2026': {'principal': 216773, 'interest_paid': 22155},
    'Jul 2026': {'principal': 199926, 'interest_paid': 23793},
    'Aug 2026': {'principal': 183340, 'interest_paid': 24500},
    'Sep 2026': {'principal': 166482, 'interest_paid': 25531},
    'Oct 2026': {'principal': 149530, 'interest_paid': 26467},
    'Nov 2026': {'principal': 132482, 'interest_paid': 27308},
    'Dec 2026': {'principal': 34538, 'interest_paid': 28054},
    'Jan 2027': {'principal': 16844, 'interest_paid': 28248},
    'Feb 2027': {'principal': 0, 'interest_paid': 28343},
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


def _get_motivational_phrase_from_cache():
    cache_data = cache.get(MOTIVATION_CACHE_KEY)
    if cache_data is not None:
        return cache_data.get('message')
    return "Error getting motivational phrase from cache"


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
        'motivational_phrase': _get_motivational_phrase_from_cache(),
    }

