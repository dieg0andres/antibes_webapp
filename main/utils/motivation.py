import json
import sys
from config.settings import LLM_API_URL, LLM_MODEL
import requests

API_URL = LLM_API_URL
DEFAULT_MODEL = LLM_MODEL
DEFAULT_PROMPT = (
    "You are a world-class motivational speaker simulator.\n\n"
    "You have a list of well-known speakers:\n"
    "- Jocko Willink\n"
    "- Jordan Peterson\n"
    "- Codie Sanchez\n"
    "- John C. Maxwell\n"
    "- David Goggins\n"
    "- Brendon Burchard\n"
    "- Brene Brown\n"
    "- Simon Sinek\n\n"
    "You also have a list of topics:\n"
    "- staying fit\n"
    "- eating healthy\n"
    "- parenting\n"
    "- corporate leadership\n"
    "- grit or resilience\n"
    "- studying\n"
    "- doing hard things\n"
    "- conscientiousness\n"
    "- homework\n"
    "- buying businesses\n"
    "- purposeful life\n\n"
    "Pick one speaker and one topic randomly.\n"
    "Then, write a short motivational message (under 28 words) in the personal tone and philosophy of that speaker, about the chosen topic.\n\n"
    "Format your answer such that it prints only the message and nothing else.\n"
)
ERROR_MESSAGE = "ERROR in generating motivation"


def generate_motivation(model: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=888)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[motivation] request failed: {exc}", file=sys.stderr)
        return ERROR_MESSAGE

    try:
        data = response.json()
    except json.JSONDecodeError:
        print("[motivation] response was not valid JSON", file=sys.stderr)
        return ERROR_MESSAGE

    if not isinstance(data, dict):
        return ERROR_MESSAGE

    if data.get("done") is False or data.get("done_reason") not in (None, "stop"):
        return ERROR_MESSAGE

    text = data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return ERROR_MESSAGE


