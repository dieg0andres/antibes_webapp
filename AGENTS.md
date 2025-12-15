# AGENTS.md

## Project context
This is a Django project (antibes_webapp). We are extracting trading-related logic into a pure-Python package:
- `options_trading/` MUST be Django-agnostic.
- Django wiring lives in `main/integrations/trading_wiring.py`.

## Current refactor goal (legacy cleanup)
Remove legacy wrapper modules once all call sites are migrated:
- main/utils/schwab_client.py
- main/utils/schwab_prices.py
- main/utils/trading_log.py

All Django code should import:
- main/integrations/trading_wiring.py for Django-aware entry points (client factory, get_client, get_latest_prices, get_trading_log_worksheet)
- options_trading.* only for pure functions that require no Django settings


## Hard constraints
- Do NOT import `django.*` or `config.settings` anywhere under `options_trading/`.
- Keep diffs minimal (no formatting-only refactors).
- Do not change business logic or output formats.
- Do not add new third-party dependencies.

## Target structure
- `options_trading/schwab/client.py`: pure `get_client(api_key, app_secret, callback_url, token_path)`
- `options_trading/schwab/manual_auth.py`: pure `run_manual_auth(api_key, app_secret, callback_url, token_path)`
- `options_trading/schwab/prices.py`: `get_latest_prices(tickers, *, client_factory, cache=None, cache_key=None, ttl=None)`
- `options_trading/sheets/trading_log.py`: injected args (spreadsheet_id, sa_key_path, worksheet_gid)
- `options_trading/wiring/env.py`: create client_factory from env vars (no Django)
- `main/integrations/trading_wiring.py`: Django wrapper that reads `django.conf.settings` and optionally `django.core.cache.cache`

## Verification commands
Run after changes:
- `python manage.py check`
- `python manage.py run_schwab_manual_auth --help`
- `python -c "from options_trading.wiring.env import make_client_factory_from_env; print('ok')"`
- rg -n "main\.utils\.(schwab_client|schwab_prices|trading_log)" .   # should return no matches

