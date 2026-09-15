from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ExaPaperSource


def test_exa_source_uses_current_exa_sdk_without_obsolete_kwargs(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    api_keys: list[str] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            api_keys.append(api_key)

        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [
                    {
                        "id": "exa-1",
                        "title": "Paper",
                        "url": "https://e.test/paper",
                        "text": "Evidence excerpt",
                        "score": 0.8,
                    }
                ],
                "costDollars": 0.03,
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
        raising=False,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            search_type="keyword",
            category="research paper",
        )
    )

    result = source.search(
        "test query",
        limit=1,
        filters=SearchPaperFilters(include_text=["evidence"], text=True),
    )

    assert result.status.status == "success"
    assert result.status.result_count == 1
    assert result.status.metadata["costDollars"] == 0.03
    assert api_keys == ["test-secret-not-for-output"]
    assert calls == [
        {
            "query": "test query",
            "type": "keyword",
            "category": "research paper",
            "contents": {
                "highlights": {"query": "test query", "max_characters": 600},
                "text": {"max_characters": 5000, "include_html_tags": False},
            },
            "num_results": 1,
            "include_text": ["evidence"],
        }
    ]
    assert "use_autoprompt" not in calls[0]
    assert result.results[0].title == "Paper"
    assert "test-secret-not-for-output" not in " ".join(result.status.errors)
