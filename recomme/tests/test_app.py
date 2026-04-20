"""
Unit tests for Flask routes in app.py.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Flask test client with mocked module-level objects."""
    import app as app_module

    mock_rec = MagicMock()
    mock_tmdb = MagicMock()
    mock_pp = MagicMock()

    # Patch the module-level instances directly so route handlers use our mocks
    with patch.object(app_module, "recommender", mock_rec), \
         patch.object(app_module, "tmdb_client", mock_tmdb), \
         patch.object(app_module, "pretty_printer", mock_pp):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c, mock_rec, mock_tmdb, mock_pp


# ---------------------------------------------------------------------------
# Test 1: GET / returns {"status": "ok"} for JSON Accept header — covers req 3.1
# ---------------------------------------------------------------------------

def test_index_returns_json_status_ok(client):
    """GET / with Accept: application/json should return {"status": "ok"}."""
    c, *_ = client
    response = c.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# Test 2: GET /recommend without movie param returns 400 — covers req 3.4
# ---------------------------------------------------------------------------

def test_recommend_missing_param_returns_400(client):
    """GET /recommend without ?movie= should return HTTP 400."""
    c, *_ = client
    response = c.get("/recommend")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


# ---------------------------------------------------------------------------
# Test 3: GET /recommend with recommender raising unexpected exception → 500
# covers req 3.5
# ---------------------------------------------------------------------------

def test_recommend_unexpected_exception_returns_500(client):
    """GET /recommend should return 500 when recommender raises an unexpected error."""
    c, mock_rec, *_ = client
    mock_rec.recommend.side_effect = RuntimeError("something went wrong")
    response = c.get("/recommend?movie=Avatar")
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data.get("error") == "internal server error"


# ---------------------------------------------------------------------------
# Test 4: GET / renders index.html — covers req 5.1
# ---------------------------------------------------------------------------

def test_index_renders_html(client):
    """GET / with default Accept should return HTML rendered from index.html."""
    c, *_ = client
    response = c.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert b"html" in response.data.lower()


# ---------------------------------------------------------------------------
# Property 5: API Returns Bounded Results for Valid Titles — covers req 3.2
# ---------------------------------------------------------------------------

# Load shared fixtures once at module level to avoid per-example overhead
import app as _app_module
from recommender import Recommender as _Recommender
from utils import PrettyPrinter as _PrettyPrinter
from urllib.parse import quote

_real_recommender = _Recommender()
_valid_titles = _real_recommender._df["title"].tolist()
_real_pp = _PrettyPrinter()


# Feature: movie-recommendation-system, Property 5: API returns bounded results for valid titles
@settings(max_examples=200, deadline=None)
@given(title=st.sampled_from(_valid_titles))
def test_api_bounded_results(title):
    """For any valid movie title, GET /recommend returns HTTP 200 with 1–10 results."""
    # Build real recommendations so we can construct a realistic mock enrich response
    recs = _real_recommender.recommend(title)

    mock_tmdb = MagicMock()
    mock_tmdb.enrich.return_value = [
        {"title": r["title"], "poster_url": None, "rating": None}
        for r in recs
    ]

    with patch.object(_app_module, "recommender", _real_recommender), \
         patch.object(_app_module, "tmdb_client", mock_tmdb), \
         patch.object(_app_module, "pretty_printer", _real_pp):
        _app_module.app.config["TESTING"] = True
        with _app_module.app.test_client() as c:
            response = c.get(f"/recommend?movie={quote(title)}")

    assert response.status_code == 200
    results = json.loads(response.data)
    assert isinstance(results, list)
    assert 1 <= len(results) <= 10


# ---------------------------------------------------------------------------
# Property 6: API Returns 404 for Unknown Titles — covers req 3.3
# ---------------------------------------------------------------------------

_known_titles_lower = set(_real_recommender._df["title_lower"].tolist())


# Feature: movie-recommendation-system, Property 6: API returns 404 for unknown titles
@settings(max_examples=200, deadline=None)
@given(title=st.text(min_size=1).filter(lambda t: t.strip().lower() not in _known_titles_lower))
def test_api_404_for_unknown(title):
    """For any string not in the dataset, GET /recommend returns HTTP 404 with a JSON error body."""
    mock_tmdb = MagicMock()

    with patch.object(_app_module, "recommender", _real_recommender), \
         patch.object(_app_module, "tmdb_client", mock_tmdb), \
         patch.object(_app_module, "pretty_printer", _real_pp):
        _app_module.app.config["TESTING"] = True
        with _app_module.app.test_client() as c:
            response = c.get(f"/recommend?movie={quote(title)}")

    assert response.status_code == 404
    data = json.loads(response.data)
    assert "error" in data


# ---------------------------------------------------------------------------
# Property 7: Case-Insensitive Lookup Consistency — covers req 3.6
# ---------------------------------------------------------------------------

# Feature: movie-recommendation-system, Property 7: Case-insensitive lookup consistency
@settings(max_examples=200, deadline=None)
@given(
    title=st.sampled_from(_valid_titles),
    transform=st.sampled_from([str.upper, str.lower, str.title]),
)
def test_case_insensitive_lookup(title, transform):
    """For any valid movie title, GET /recommend returns the same set of recommended
    titles regardless of the casing variant (upper, lower, title-case) used in the query.

    Validates: Requirements 3.6
    """
    transformed_title = transform(title)

    # Build canonical recommendations using the original title
    canonical_recs = _real_recommender.recommend(title)
    canonical_titles = {r["title"] for r in canonical_recs}

    mock_tmdb = MagicMock()
    mock_tmdb.enrich.side_effect = lambda titles: [
        {"title": t, "poster_url": None, "rating": None} for t in titles
    ]

    with patch.object(_app_module, "recommender", _real_recommender), \
         patch.object(_app_module, "tmdb_client", mock_tmdb), \
         patch.object(_app_module, "pretty_printer", _real_pp):
        _app_module.app.config["TESTING"] = True
        with _app_module.app.test_client() as c:
            response = c.get(f"/recommend?movie={quote(transformed_title)}")

    assert response.status_code == 200
    results = json.loads(response.data)
    result_titles = {r["title"] for r in results}
    assert result_titles == canonical_titles
