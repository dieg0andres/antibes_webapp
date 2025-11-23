from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

try:
    from schwab.auth import client_from_manual_flow
except ImportError as exc:  # pragma: no cover - dependency optional until installed
    client_from_manual_flow = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


class Command(BaseCommand):
    help = "Run Schwab manual OAuth flow and write token to SCHWAB_TOKEN_PATH"

    def handle(self, *args, **options):
        if client_from_manual_flow is None:
            raise RuntimeError(
                "schwab-py must be installed to run the Schwab manual auth command."
            ) from _IMPORT_ERROR

        token_path = Path(settings.SCHWAB_TOKEN_PATH)
        token_path.parent.mkdir(parents=True, exist_ok=True)

        client_from_manual_flow(
            api_key=settings.SCHWAB_API_KEY,
            app_secret=settings.SCHWAB_APP_SECRET,
            callback_url=settings.SCHWAB_CALLBACK_URL,
            token_path=str(token_path),
        )

        self.stdout.write(self.style.SUCCESS(f"Token written to {token_path}"))

