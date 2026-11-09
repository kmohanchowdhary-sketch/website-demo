"""Unit tests for ContentBasedRecommender."""

from __future__ import annotations

import pandas as pd
import pytest

from src.content_based import ContentBasedRecommender


@pytest.fixture
def sample_catalog():
    movies = pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story (1995)",
                "clean_title": "Toy Story",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "year": 1995,
                "metadata_soup": "adventure animation children comedy fantasy adventure animation children comedy fantasy pixar fun",
            },
            {
                "movieId": 2,
                "title": "Jumanji (1995)",
                "clean_title": "Jumanji",
                "genres": "Adventure|Children|Fantasy",
                "year": 1995,
                "metadata_soup": "adventure children fantasy adventure children fantasy board game",
            },
            {
                "movieId": 3,
                "title": "Heat (1995)",
                "clean_title": "Heat",
                "genres": "Action|Crime|Thriller",
                "year": 1995,
                "metadata_soup": "action crime thriller action crime thriller heist cops",
            },
            {
                "movieId": 4,
                "title": "Toy Story 2 (1999)",
                "clean_title": "Toy Story 2",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "year": 1999,
                "metadata_soup": "adventure animation children comedy fantasy adventure animation children comedy fantasy pixar toys",
            },
            {
                "movieId": 5,
                "title": "Casino (1995)",
                "clean_title": "Casino",
                "genres": "Crime|Drama",
                "year": 1995,
                "metadata_soup": "crime drama crime drama mafia vegas",
            },
        ]
    )
    return movies


def test_content_recommender_initialization(sample_catalog):
    cb = ContentBasedRecommender(sample_catalog)
    assert cb.tfidf_matrix.shape[0] == 5
    assert 1 in cb.movie_id_to_idx


def test_recommend_for_movie(sample_catalog):
    cb = ContentBasedRecommender(sample_catalog)
    # Recommend similar to Toy Story (movieId 1)
    recs = cb.recommend_for_movie(1, top_n=2)

    assert not recs.empty
    # Toy Story itself should NOT be in recommendations
    assert 1 not in recs["movieId"].values
    # Toy Story 2 (movieId 4) and Jumanji (movieId 2) should rank high
    top_mid = recs.iloc[0]["movieId"]
    assert top_mid in (4, 2)
    assert "explanation" in recs.columns
    assert "content_score" in recs.columns


def test_recommend_for_user(sample_catalog):
    cb = ContentBasedRecommender(sample_catalog)

    # User who loved Heat (movieId 3) with 5.0 rating
    user_ratings = pd.DataFrame([{"movieId": 3, "rating": 5.0}])

    recs = cb.recommend_for_user(user_id=1, user_ratings_df=user_ratings, top_n=3)

    assert not recs.empty
    # Already rated Heat should be excluded
    assert 3 not in recs["movieId"].values
    # Casino (movieId 5) also shares Crime, should rank above Toy Story
    assert 5 in recs["movieId"].values


def test_recommend_for_custom_preferences(sample_catalog):
    cb = ContentBasedRecommender(sample_catalog)

    recs = cb.recommend_for_custom_preferences(
        favorite_movie_ids=[1],
        preferred_genres=["Animation"],
        top_n=2,
    )

    assert not recs.empty
    assert 1 not in recs["movieId"].values
    assert recs.iloc[0]["movieId"] == 4  # Toy Story 2 is top match
