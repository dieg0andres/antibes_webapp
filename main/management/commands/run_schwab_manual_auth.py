from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from options_trading.schwab.manual_auth import run_manual_auth


class Command(BaseCommand):
    help = "Run Schwab manual OAuth flow and write token to SCHWAB_TOKEN_PATH"

    def handle(self, *args, **options):
        token_path = Path(settings.SCHWAB_TOKEN_PATH)
        token_path.parent.mkdir(parents=True, exist_ok=True)

        run_manual_auth(
            api_key=settings.SCHWAB_API_KEY,
            app_secret=settings.SCHWAB_APP_SECRET,
            callback_url=settings.SCHWAB_CALLBACK_URL,
            token_path=token_path,
        )

        self.stdout.write(self.style.SUCCESS(f"Token written to {token_path}"))

