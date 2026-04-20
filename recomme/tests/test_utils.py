"""
Unit tests for utils.py — TMDBClient and PrettyPrinter.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Test: TMDBClient raises EnvironmentError when TMDB_API_KEY is not set
# covers requirement 6.5
# ---------------------------------------------------------------------------

def test_tmdb_client_raises_without_api_key(monkeypatch):
    """TMDBClient should raise EnvironmentError when TMDB_API_KEY env var is absent."""
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    # Import after env manipulation so the constructor reads the patched env
    from utils import TMDBClient
    with pytest.raises(EnvironmentError):
        TMDBClient()


# ---------------------------------------------------------------------------
# Property 8: TMDB Enrich Returns Required Keys
# Feature: movie-recommendation-system, Property 8: TMDB enrich returns required keys
# ---------------------------------------------------------------------------

def _make_mock_response(poster_path="/abc.jpg", vote_average=7.5):
    """Helper to build a mock requests.Response for TMDB search."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "results": [{"poster_path": poster_path, "vote_average": vote_average}]
    }
    return mock_resp


# Feature: movie-recommendation-system, Property 8: TMDB enrich returns required keys
@given(titles=st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=20))
@settings(max_examples=200, deadline=None)
def test_tmdb_enrich_keys(titles):
    """Validates: Requirements 6.1
    For any list of title strings, enrich() returns a list of the same length
    where every entry contains the keys title, poster_url, and rating.
    """
    from utils import TMDBClient

    with patch("utils.requests.get", return_value=_make_mock_response()):
        client = TMDBClient(api_key="fake-key")
        results = client.enrich(titles)

    assert len(results) == len(titles)
    for entry in results:
        assert "title" in entry
        assert "poster_url" in entry
        assert "rating" in entry


# ---------------------------------------------------------------------------
# Property 9: Serialization Round-Trip
# Feature: movie-recommendation-system, Property 9: Serialization round-trip
# ---------------------------------------------------------------------------

# Feature: movie-recommendation-system, Property 9: Serialization round-trip
@given(
    recommendations=st.lists(
        st.fixed_dictionaries({
            "title": st.text(min_size=1),
            "similarity_score": st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False
            ),
        }),
        min_size=0,
        max_size=10,
    )
)
@settings(max_examples=200, deadline=None)
def test_serialization_round_trip(recommendations):
    """Validates: Requirements 8.1, 8.2, 8.3
    For any valid list of recommendation result dicts, serializing then
    deserializing then re-serializing shall produce an identical JSON string.
    """
    from utils import PrettyPrinter
    pp = PrettyPrinter()
    first = pp.serialize(recommendations)
    assert pp.serialize(pp.deserialize(first)) == first
