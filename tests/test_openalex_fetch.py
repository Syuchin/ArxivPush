import unittest

import requests

import daily_paper


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class OpenAlexFetchTests(unittest.TestCase):
    def test_free_mode_uses_api_key_query_param(self):
        params = daily_paper._build_openalex_params(
            query="agent",
            max_results=5,
            since_hours=24,
            email="test@example.com",
            api_key="secret-key",
            use_premium=False,
        )

        self.assertEqual(params["api_key"], "secret-key")
        self.assertEqual(params["sort"], "publication_date:desc")
        self.assertIn("from_publication_date:", params["filter"])
        self.assertNotIn("from_created_date:", params["filter"])

    def test_premium_plan_upgrade_falls_back_to_free_mode(self):
        premium_error = FakeResponse(
            status_code=429,
            payload={
                "error": "Plan upgrade required",
                "message": "The from_created_date filter requires a Premium plan.",
            },
            text="Plan upgrade required",
        )
        success = FakeResponse(
            payload={
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Test paper",
                        "publication_date": "2026-03-28",
                        "ids": {"arxiv": "https://arxiv.org/abs/2603.12345"},
                        "abstract": "summary",
                    }
                ],
            }
        )
        session = RecordingSession([premium_error, success])

        papers = daily_paper.fetch_papers_from_openalex(
            query="agent",
            max_results=1,
            since_hours=24,
            email="test@example.com",
            session=session,
            api_key="secret-key",
            use_premium=True,
        )

        self.assertEqual(len(papers), 1)
        self.assertEqual(len(session.calls), 2)
        first_params = session.calls[0][1]["params"]
        second_params = session.calls[1][1]["params"]
        self.assertIn("from_created_date:", first_params["filter"])
        self.assertIn("from_publication_date:", second_params["filter"])
        self.assertEqual(first_params["api_key"], "secret-key")
        self.assertEqual(second_params["api_key"], "secret-key")
        self.assertNotIn("headers", session.calls[0][1])
        self.assertNotIn("headers", session.calls[1][1])


if __name__ == "__main__":
    unittest.main()
