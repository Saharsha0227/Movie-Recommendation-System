"""
Unit tests and property-based tests for the Recommender class.
"""
import os
import tempfile
import pytest
import pandas as pd
from hypothesis import given, settings, strategies as st
from recommender import Recommender, MovieNotFoundError


# ---------------------------------------------------------------------------
# Fixture: shared Recommender instance (loaded once per session for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def recommender():
    """Load the real CSV once for the whole test session."""
    return Recommender("tmdb_5000_movies.csv")


# ---------------------------------------------------------------------------
# Test 1: Successful load from real CSV (smoke test) — covers requirement 1.1
# ---------------------------------------------------------------------------

def test_load_from_real_csv(recommender):
    """Recommender should load without error and expose a non-empty dataframe."""
    assert recommender._df is not None
    assert len(recommender._df) > 0


# ---------------------------------------------------------------------------
# Test 2: FileNotFoundError for missing path — covers requirement 1.5
# ---------------------------------------------------------------------------

def test_file_not_found_error():
    """Recommender should raise FileNotFoundError for a non-existent CSV path."""
    with pytest.raises(FileNotFoundError):
        Recommender("non_existent_path_xyz.csv")


# ---------------------------------------------------------------------------
# Test 3: CountVectorizer configured correctly — covers requirement 2.1
# ---------------------------------------------------------------------------

def test_vectorizer_configuration(recommender):
    """CountVectorizer must use max_features=5000 and stop_words='english'."""
    cv = recommender._vectorizer
    assert cv.max_features == 5000
    assert cv.stop_words == "english"


# ---------------------------------------------------------------------------
# Test 4: Similarity matrix is same object across calls — covers requirement 2.3
# ---------------------------------------------------------------------------

def test_similarity_matrix_cached(recommender):
    """The similarity matrix should be the same object on repeated recommend() calls."""
    matrix_before = id(recommender._similarity)
    recommender.recommend("Avatar")
    matrix_after = id(recommender._similarity)
    assert matrix_before == matrix_after


# ---------------------------------------------------------------------------
# Helper: write a synthetic DataFrame to a temp CSV and run preprocessing
# ---------------------------------------------------------------------------

def _run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Write df to a temp CSV, instantiate Recommender, return the internal df."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    try:
        r = Recommender(dataset_path=tmp_path)
        return r._df
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 1: Preprocessing Invariants
# Validates: Requirements 1.2, 1.3, 1.4
# ---------------------------------------------------------------------------

# Hypothesis strategy: generate a non-empty list of rows with the required columns.
# Each cell is either a short text string or None (to exercise null-dropping).
_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=1,
    max_size=30,
)
_nullable_text = st.one_of(st.none(), _text)

_row_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=1, max_value=999999),
        "title": _text,
        "overview": _nullable_text,
        "genres": _nullable_text,
        "keywords": _nullable_text,
        "cast": _nullable_text,
        "crew": _nullable_text,
    }
)


# Feature: movie-recommendation-system, Property 1: Preprocessing invariants
@given(rows=st.lists(_row_strategy, min_size=1, max_size=50))
@settings(max_examples=200, deadline=None)
def test_preprocessing_invariants(rows):
    """
    For any CSV input with the required columns, after preprocessing:
    (a) the result contains the columns id, title, overview, genres, keywords,
        cast, crew, and tags;
    (b) no null values exist in any of those columns;
    (c) each row's tags field contains the content of overview, genres,
        keywords, cast, and crew.
    """
    raw_df = pd.DataFrame(rows)

    # Rows where any text column is None will be dropped by the Recommender.
    text_cols = ["overview", "genres", "keywords", "cast", "crew"]
    complete_rows = raw_df.dropna(subset=text_cols)

    # If all rows have nulls, preprocessing produces an empty df — skip.
    if complete_rows.empty:
        return

    result = _run_preprocessing(raw_df)

    # (a) Required columns are present
    required_cols = {"id", "title", "overview", "genres", "keywords", "cast", "crew", "tags"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )

    # (b) No null values in any required column
    for col in required_cols:
        assert result[col].notna().all(), f"Column '{col}' contains null values after preprocessing"

    # (c) tags contains the content of each source text column
    for _, row in result.iterrows():
        tags = row["tags"]
        for col in text_cols:
            assert str(row[col]) in tags, (
                f"tags field missing content from '{col}': "
                f"expected '{row[col]}' to appear in '{tags}'"
            )


# ---------------------------------------------------------------------------
# Property 2: Similarity Matrix Mathematical Invariants
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

# A word is at least 2 ASCII letters (matches CountVectorizer's default token pattern).
_word = st.from_regex(r"[a-zA-Z]{2,10}", fullmatch=True)
_tag = st.lists(_word, min_size=1, max_size=10).map(" ".join)


# Feature: movie-recommendation-system, Property 2: Similarity matrix invariants
@given(corpus=st.lists(_tag, min_size=2, max_size=20))
@settings(max_examples=200, deadline=None)
def test_similarity_matrix_invariants(corpus):
    """
    For any preprocessed dataset (represented here as a small tag corpus),
    the computed similarity matrix shall:
    (a) have all values in the range [0, 1];
    (b) have a diagonal where every entry equals 1.0.
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cv = CountVectorizer(max_features=5000, stop_words="english")
    try:
        vectors = cv.fit_transform(corpus)
    except ValueError:
        # All tokens were stop words — no features; skip this example.
        return

    if vectors.shape[1] == 0:
        return

    # Skip any corpus where a document has an all-zero vector (all tokens were
    # stop words), since cosine similarity is undefined (0/0) for such vectors.
    row_norms = np.asarray(vectors.sum(axis=1)).ravel()
    if (row_norms == 0).any():
        return

    matrix = cosine_similarity(vectors)

    # (a) All values must be in [0, 1]
    assert matrix.min() >= 0.0 - 1e-9, (
        f"Similarity matrix contains value below 0: {matrix.min()}"
    )
    assert matrix.max() <= 1.0 + 1e-9, (
        f"Similarity matrix contains value above 1: {matrix.max()}"
    )

    # (b) Diagonal must be 1.0 for every entry
    diag = np.diag(matrix)
    assert np.allclose(diag, 1.0), (
        f"Diagonal entries are not all 1.0: {diag}"
    )


# ---------------------------------------------------------------------------
# Property 3: Recommendation Result Invariants
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------

# Feature: movie-recommendation-system, Property 3: Recommendation result invariants
@given(title=st.data())
@settings(max_examples=200, deadline=None)
def test_recommendation_result_invariants(recommender, title):
    """
    For any valid movie title in the dataset, calling recommend(title) shall:
    (a) return a list of length at most 10;
    (b) contain no entry whose title equals the queried title;
    (c) be ordered by descending similarity_score.
    """
    # Draw a random title from the loaded dataset
    known_titles = recommender._df["title"].tolist()
    chosen_title = title.draw(st.sampled_from(known_titles))

    results = recommender.recommend(chosen_title)

    # (a) At most 10 results
    assert len(results) <= 10, (
        f"recommend('{chosen_title}') returned {len(results)} results, expected ≤ 10"
    )

    # (b) No self-recommendation
    chosen_lower = chosen_title.lower()
    for entry in results:
        assert entry["title"].lower() != chosen_lower, (
            f"recommend('{chosen_title}') returned the queried movie itself: {entry['title']}"
        )

    # (c) Descending order by similarity_score
    scores = [entry["similarity_score"] for entry in results]
    assert scores == sorted(scores, reverse=True), (
        f"recommend('{chosen_title}') results are not in descending order: {scores}"
    )


# ---------------------------------------------------------------------------
# Property 4: MovieNotFoundError for Unknown Titles
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

# Feature: movie-recommendation-system, Property 4: MovieNotFoundError for unknown titles
@given(title=st.text())
@settings(max_examples=200, deadline=None)
def test_movie_not_found_error(recommender, title):
    """
    For any string that is not present in the dataset's title index,
    calling recommend(title) shall raise MovieNotFoundError.
    """
    known_titles_lower = set(recommender._df["title_lower"].tolist())

    # Only test strings that are genuinely not in the dataset
    if title.lower() in known_titles_lower:
        return

    with pytest.raises(MovieNotFoundError):
        recommender.recommend(title)
