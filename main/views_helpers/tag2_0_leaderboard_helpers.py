from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone


LEADERBOARD_FILENAME = "tag2_0_leaderboard.json"
MAX_LEADERBOARD_ENTRIES = 10
MAX_NAME_LENGTH = 20
MAX_MESSAGE_LENGTH = 50
EMPTY_LEADERBOARD = {"top_10_scores": []}


class LeaderboardRequestError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _leaderboard_path() -> Path:
    return Path(settings.BASE_DIR) / "data_store" / LEADERBOARD_FILENAME


def _normalize_leaderboard(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return dict(EMPTY_LEADERBOARD)

    scores = payload.get("top_10_scores")
    if not isinstance(scores, list):
        return dict(EMPTY_LEADERBOARD)

    valid_scores = [score for score in scores if _is_saved_score(score)]
    return {"top_10_scores": _sort_scores(valid_scores)[:MAX_LEADERBOARD_ENTRIES]}


def _read_leaderboard() -> dict[str, list[dict[str, Any]]]:
    leaderboard_path = _leaderboard_path()
    if not leaderboard_path.is_file():
        return dict(EMPTY_LEADERBOARD)

    try:
        with leaderboard_path.open("r", encoding="utf-8") as leaderboard_file:
            return _normalize_leaderboard(json.load(leaderboard_file))
    except (json.JSONDecodeError, OSError):
        return dict(EMPTY_LEADERBOARD)


def _write_leaderboard(payload: dict[str, list[dict[str, Any]]]) -> None:
    leaderboard_path = _leaderboard_path()
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = leaderboard_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as leaderboard_file:
        json.dump(payload, leaderboard_file, indent=2)
        leaderboard_file.write("\n")
    temp_path.replace(leaderboard_path)


def _sort_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        scores,
        key=lambda score: score["score"],
        reverse=True,
    )


def _is_saved_score(score: Any) -> bool:
    if not isinstance(score, dict):
        return False

    score_value = score.get("score")
    return isinstance(score_value, int) and not isinstance(score_value, bool)


def _parse_json_body(request) -> dict[str, Any]:
    content_type = request.content_type or ""
    if "application/json" not in content_type.lower():
        raise LeaderboardRequestError("POST requests must use application/json.")

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LeaderboardRequestError("POST body must be valid JSON.")

    if not isinstance(payload, dict):
        raise LeaderboardRequestError("POST body must be a JSON object.")

    return payload


def _validate_submission(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LeaderboardRequestError("name is required and must be text.")

    name = name.strip()
    if len(name) > MAX_NAME_LENGTH:
        raise LeaderboardRequestError(
            f"name must be {MAX_NAME_LENGTH} characters or fewer."
        )

    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        raise LeaderboardRequestError("score is required and must be an integer.")

    submission = {
        "date": timezone.localtime().isoformat(),
        "name": name,
        "score": score,
    }

    message = payload.get("message")
    if message is not None:
        if not isinstance(message, str):
            raise LeaderboardRequestError("message must be text.")

        message = message.strip()
        if len(message) > MAX_MESSAGE_LENGTH:
            raise LeaderboardRequestError(
                f"message must be {MAX_MESSAGE_LENGTH} characters or fewer."
            )

        if message:
            submission["message"] = message

    return submission


def get_leaderboard() -> dict[str, list[dict[str, Any]]]:
    return _read_leaderboard()


def submit_leaderboard_score(request) -> dict[str, Any]:
    submission = _validate_submission(_parse_json_body(request))
    leaderboard = _read_leaderboard()
    scores = leaderboard["top_10_scores"]

    is_full = len(scores) >= MAX_LEADERBOARD_ENTRIES
    lowest_top_score = min((score["score"] for score in scores), default=None)
    if (
        is_full
        and lowest_top_score is not None
        and submission["score"] <= lowest_top_score
    ):
        return {
            "saved": False,
            "message": "Score was not high enough for the top 10.",
            "leaderboard": leaderboard,
        }

    updated_leaderboard = {
        "top_10_scores": _sort_scores([*scores, submission])[:MAX_LEADERBOARD_ENTRIES]
    }
    _write_leaderboard(updated_leaderboard)

    return {
        "saved": True,
        "message": "Score saved.",
        "leaderboard": updated_leaderboard,
    }
