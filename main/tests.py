import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import TestCase, override_settings
from django.urls import reverse


class TagLeaderboardTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.override_settings = override_settings(BASE_DIR=Path(self.temp_dir.name))
        self.override_settings.enable()
        self.url = reverse("tag2_0_leaderboard")

    def tearDown(self):
        self.override_settings.disable()
        self.temp_dir.cleanup()

    @property
    def leaderboard_path(self):
        return Path(self.temp_dir.name) / "data_store" / "tag2_0_leaderboard.json"

    def post_json(self, payload):
        return self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_get_returns_empty_leaderboard_without_creating_file(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"top_10_scores": []})
        self.assertFalse(self.leaderboard_path.exists())

    def test_post_requires_json_content_type(self):
        response = self.client.post(self.url, data={"name": "Diego", "score": 10})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")
        self.assertFalse(self.leaderboard_path.exists())

    def test_post_validates_required_json_fields(self):
        response = self.post_json({"name": "Diego"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")
        self.assertFalse(self.leaderboard_path.exists())

    def test_post_saves_valid_score(self):
        response = self.post_json(
            {"name": "Diego", "score": 42, "message": "Nice run"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["saved"])

        with self.leaderboard_path.open(encoding="utf-8") as leaderboard_file:
            saved_leaderboard = json.load(leaderboard_file)

        self.assertEqual(len(saved_leaderboard["top_10_scores"]), 1)
        saved_score = saved_leaderboard["top_10_scores"][0]
        self.assertEqual(saved_score["name"], "Diego")
        self.assertEqual(saved_score["score"], 42)
        self.assertEqual(saved_score["message"], "Nice run")
        self.assertIn("date", saved_score)

    def test_leaderboard_keeps_top_10_scores_only(self):
        for score in range(1, 11):
            response = self.post_json({"name": f"Player {score}", "score": score})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["saved"])

        low_score_response = self.post_json({"name": "Low", "score": 0})
        self.assertEqual(low_score_response.status_code, 200)
        self.assertFalse(low_score_response.json()["saved"])

        high_score_response = self.post_json({"name": "High", "score": 11})
        self.assertEqual(high_score_response.status_code, 200)
        self.assertTrue(high_score_response.json()["saved"])

        saved_scores = high_score_response.json()["leaderboard"]["top_10_scores"]
        score_values = [score["score"] for score in saved_scores]

        self.assertEqual(len(saved_scores), 10)
        self.assertEqual(score_values, list(range(11, 1, -1)))
