"""
recommender.py — ML Engine
Handles dataset loading, preprocessing, and content-based movie recommendations.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieNotFoundError(Exception):
    """Raised when a title is not present in the dataset."""


class Recommender:
    def __init__(self, dataset_path: str = "tmdb_5000_movies.csv") -> None:
        """Load dataset, build vectorizer and similarity matrix at construction time.
        Raises FileNotFoundError with a descriptive message if dataset_path is missing."""

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. "
                "Please ensure tmdb_5000_movies.csv is present in the working directory."
            )

        df = pd.read_csv(dataset_path)

        # Attempt to merge with credits file if available (provides cast/crew columns)
        credits_path = os.path.join(os.path.dirname(dataset_path), "tmdb_5000_credits.csv")
        if os.path.exists(credits_path):
            credits = pd.read_csv(credits_path)
            df = df.merge(credits, on="title")

        # Determine which text columns are available
        text_cols = ["overview", "genres", "keywords", "cast", "crew"]
        available_text_cols = [c for c in text_cols if c in df.columns]

        # Select only the columns we need
        id_col = "movie_id" if "movie_id" in df.columns else "id"
        keep_cols = [id_col, "title"] + available_text_cols
        df = df[keep_cols]
        if id_col != "id":
            df = df.rename(columns={id_col: "id"})

        df = df.dropna()
        df = df.reset_index(drop=True)

        tag_parts = [df[c].astype(str) for c in available_text_cols]
        df["tags"] = tag_parts[0]
        for part in tag_parts[1:]:
            df["tags"] = df["tags"] + " " + part

        df["title_lower"] = df["title"].str.lower()

        self._df = df

        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(df["tags"])
        self._vectorizer = cv

        self._similarity = cosine_similarity(vectors)


    def recommend(self, title: str, top_n: int = 10) -> list[dict]:
        """Return up to top_n recommendations for title (case-insensitive).
        Each dict has keys: title (str), similarity_score (float).
        Raises MovieNotFoundError if title is not in the dataset."""
        query = title.lower()
        matches = self._df[self._df["title_lower"] == query]
        if matches.empty:
            raise MovieNotFoundError(f"'{title}' not found in dataset")

        idx = matches.index[0]
        scores = list(enumerate(self._similarity[idx]))
        # Exclude the queried movie itself and sort by descending score
        scores = [(i, s) for i, s in scores if i != idx]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"title": self._df.iloc[i]["title"], "similarity_score": float(score)}
            for i, score in scores[:top_n]
        ]

