"""Unit tests for CollaborativeFilteringRecommender."""

from __future__ import annotations

import pandas as pd
import pytest

from src.collaborative import CollaborativeFilteringRecommender


@pytest.fixture
def sample_cf_data():
    movies_df = pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story (1995)",
                "clean_title": "Toy Story",
                "genres": "Animation|Children|Comedy",
                "year": 1995,
            },
            {
                "movieId": 2,
                "title": "Jumanji (1995)",
                "clean_title": "Jumanji",
                "genres": "Adventure|Children",
                "year": 1995,
            },
            {
                "movieId": 3,
                "title": "Grumpier Old Men (1995)",
                "clean_title": "Grumpier Old Men",
                "genres": "Comedy|Romance",
                "year": 1995,
            },
            {
                "movieId": 4,
                "title": "Waiting to Exhale (1995)",
                "clean_title": "Waiting to Exhale",
                "genres": "Comedy|Drama",
                "year": 1995,
            },
            {
                "movieId": 5,
                "title": "Father of the Bride Part II (1995)",
                "clean_title": "Father of the Bride Part II",
                "genres": "Comedy",
                "year": 1995,
            },
        ]
    )

    ratings_df = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0},
            {"userId": 1, "movieId": 2, "rating": 4.0},
            {"userId": 1, "movieId": 3, "rating": 3.0},
            {"userId": 2, "movieId": 1, "rating": 4.5},
            {"userId": 2, "movieId": 2, "rating": 4.0},
            {"userId": 2, "movieId": 4, "rating": 2.0},
            {"userId": 3, "movieId": 1, "rating": 5.0},
            {"userId": 3, "movieId": 3, "rating": 3.5},
            {"userId": 3, "movieId": 5, "rating": 4.0},
            {"userId": 4, "movieId": 2, "rating": 4.5},
            {"userId": 4, "movieId": 3, "rating": 3.0},
            {"userId": 4, "movieId": 5, "rating": 4.0},
        ]
    )
    return movies_df, ratings_df


def test_cf_initialization(sample_cf_data):
    movies_df, ratings_df = sample_cf_data
    cf = CollaborativeFilteringRecommender(
        ratings_df=ratings_df,
        movies_df=movies_df,
        min_movie_ratings=1,
    )
    assert cf.user_item_matrix.shape[0] == 4  # 4 users
    assert cf.user_item_matrix.shape[1] == 5  # 5 movies
    assert cf.item_similarity_matrix.shape == (5, 5)


def test_cf_predict_rating(sample_cf_data):
    movies_df, ratings_df = sample_cf_data
    cf = CollaborativeFilteringRecommender(
        ratings_df=ratings_df,
        movies_df=movies_df,
        min_movie_ratings=1,
    )

    # Predict user 1's rating for unrated movie 5
    pred, contributors = cf.predict_rating(user_id=1, movie_id=5)
    assert 0.5 <= pred <= 5.0


def test_cf_recommend_for_user_excludes_rated(sample_cf_data):
    movies_df, ratings_df = sample_cf_data
    cf = CollaborativeFilteringRecommender(
        ratings_df=ratings_df,
        movies_df=movies_df,
        min_movie_ratings=1,
    )

    # User 1 already rated movies 1, 2, 3
    recs = cf.recommend_for_user(user_id=1, top_n=5)
    assert not recs.empty
    rec_ids = set(recs["movieId"])
    assert 1 not in rec_ids
    assert 2 not in rec_ids
    assert 3 not in rec_ids
    assert "predicted_rating" in recs.columns
    assert "explanation" in recs.columns


def test_cf_cold_start_handling(sample_cf_data):
    movies_df, ratings_df = sample_cf_data
    cf = CollaborativeFilteringRecommender(
        ratings_df=ratings_df,
        movies_df=movies_df,
        min_movie_ratings=1,
    )

    # Brand new user with ID 9999
    recs = cf.recommend_for_user(user_id=9999, top_n=3)
    assert not recs.empty
    assert len(recs) <= 3
    assert "Cold-start fallback" in recs.iloc[0]["explanation"]
