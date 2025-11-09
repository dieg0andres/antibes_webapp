from django.conf import settings
from django.core.cache import cache
from django.core.management.base import BaseCommand, CommandError
import logging

from main.utils.motivation import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    ERROR_MESSAGE,
    generate_motivation,
)
from config.settings import MOTIVATION_CACHE_KEY, MOTIVATION_CACHE_TTL

LOG = logging.getLogger(__name__)

DEFAULT_CACHE_KEY = MOTIVATION_CACHE_KEY
DEFAULT_CACHE_TTL = int(MOTIVATION_CACHE_TTL)


class Command(BaseCommand):
    help = "Generate motivation text using the local LLM service."

    def add_arguments(self, parser):
        parser.add_argument("--model", help="Model identifier to use")
        parser.add_argument("--prompt", help="Prompt text to send to the model")

    def handle(self, *args, **options):
        model = options.get("model") or DEFAULT_MODEL
        prompt = options.get("prompt") or DEFAULT_PROMPT

        LOG.debug("Generating motivation with model=%s", model)
        message = generate_motivation(model=model, prompt=prompt)


        cache_payload = {
            "ok": message != ERROR_MESSAGE,
            "message": message,
        }
        cache.set(DEFAULT_CACHE_KEY, cache_payload, timeout=DEFAULT_CACHE_TTL)

        if message == ERROR_MESSAGE:
            LOG.error("Motivation generation failed for model=%s", model)
            raise CommandError(ERROR_MESSAGE)

        self.stdout.write(self.style.SUCCESS(message))

