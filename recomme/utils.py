"""
utils.py — Shared helpers
Contains TMDBClient for poster/rating enrichment and PrettyPrinter for JSON serialization.
"""
import json
import os
import requests


TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


class TMDBClient:
    def __init__(self, api_key: str = None) -> None:
        if not api_key:
            api_key = os.environ.get("TMDB_API_KEY")
        self.api_key = api_key  # None means no enrichment, poster_url/rating will be null

    def enrich(self, titles: list[str]) -> list[dict]:
        """For each title fetch poster_url (str | None) and rating (float | None).
        Returns list of dicts: {title, poster_url, rating}."""
        results = []
        for title in titles:
            poster_url = None
            rating = None
            if not self.api_key:
                results.append({"title": title, "poster_url": None, "rating": None})
                continue
            try:
                response = requests.get(
                    TMDB_SEARCH_URL,
                    params={"api_key": self.api_key, "query": title},
                    timeout=5,
                )
                response.raise_for_status()
                data = response.json()
                movies = data.get("results", [])
                if movies:
                    first = movies[0]
                    poster_path = first.get("poster_path")
                    if poster_path:
                        poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
                    vote = first.get("vote_average")
                    if vote is not None:
                        rating = float(vote)
            except Exception:
                poster_url = None
                rating = None
            results.append({"title": title, "poster_url": poster_url, "rating": rating})
        return results


class PrettyPrinter:
    def serialize(self, recommendations: list[dict]) -> str:
        """Serialize a list of recommendation dicts to a canonical JSON string.
        Each dict must contain at minimum 'title' and 'similarity_score'."""
        return json.dumps(recommendations, sort_keys=True)

    def deserialize(self, json_str: str) -> list[dict]:
        """Deserialize a JSON string back to a list of recommendation dicts."""
        return json.loads(json_str)
