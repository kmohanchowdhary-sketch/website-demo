"""Unit tests for HybridRecommendationAgent."""

from __future__ import annotations

import pandas as pd
import pytest

from src.collaborative import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommendationAgent


@pytest.fixture
def hybrid_setup():
    movies_df = pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story (1995)",
                "clean_title": "Toy Story",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "year": 1995,
                "metadata_soup": "adventure animation children comedy fantasy adventure animation children comedy fantasy pixar",
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

    ratings_df = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0},
            {"userId": 1, "movieId": 2, "rating": 4.5},
            {"userId": 2, "movieId": 1, "rating": 4.5},
            {"userId": 2, "movieId": 4, "rating": 5.0},
            {"userId": 3, "movieId": 3, "rating": 5.0},
            {"userId": 3, "movieId": 5, "rating": 4.5},
            {"userId": 4, "movieId": 1, "rating": 4.0},
            {"userId": 4, "movieId": 4, "rating": 4.5},
        ]
    )

    cb = ContentBasedRecommender(movies_df)
    cf = CollaborativeFilteringRecommender(ratings_df, movies_df, min_movie_ratings=1)
    agent = HybridRecommendationAgent(cb, cf)

    return agent, movies_df, ratings_df


def test_weight_validation(hybrid_setup):
    agent, _, _ = hybrid_setup

    # Valid weights normalize to 1.0
    w_cb, w_cf = agent.validate_and_normalize_weights(0.6, 0.4)
    assert abs((w_cb + w_cf) - 1.0) < 1e-4

    # Different scale
    w_cb, w_cf = agent.validate_and_normalize_weights(3, 7)
    assert w_cb == 0.3
    assert w_cf == 0.7

    # Both zero defaults to 0.5, 0.5
    w_cb, w_cf = agent.validate_and_normalize_weights(0, 0)
    assert w_cb == 0.5 and w_cf == 0.5

    # Negative weights raise error
    with pytest.raises(ValueError):
        agent.validate_and_normalize_weights(-0.5, 0.5)


def test_hybrid_recommend_for_user(hybrid_setup):
    agent, _, ratings_df = hybrid_setup

    u1_ratings = ratings_df[ratings_df["userId"] == 1]
    recs = agent.recommend_for_user(
        user_id=1,
        user_ratings_df=u1_ratings,
        content_weight=0.5,
        collab_weight=0.5,
        top_n=3,
    )

    assert not recs.empty
    # User 1 rated 1 and 2, so 1 and 2 must not be in recommendations
    rec_ids = set(recs["movieId"])
    assert 1 not in rec_ids
    assert 2 not in rec_ids

    # Check required columns
    assert "hybrid_score" in recs.columns
    assert "content_score" in recs.columns
    assert "collab_score" in recs.columns
    assert "contribution_type" in recs.columns
    assert "explanation" in recs.columns


def test_hybrid_pure_content_mode(hybrid_setup):
    agent, _, ratings_df = hybrid_setup

    u1_ratings = ratings_df[ratings_df["userId"] == 1]
    recs = agent.recommend_for_user(
        user_id=1,
        user_ratings_df=u1_ratings,
        content_weight=1.0,
        collab_weight=0.0,
        top_n=2,
    )

    assert not recs.empty
    assert recs.iloc[0]["contribution_type"] == "Content-Based Only"


def test_hybrid_pure_collaborative_mode(hybrid_setup):
    agent, _, ratings_df = hybrid_setup

    u1_ratings = ratings_df[ratings_df["userId"] == 1]
    recs = agent.recommend_for_user(
        user_id=1,
        user_ratings_df=u1_ratings,
        content_weight=0.0,
        collab_weight=1.0,
        top_n=2,
    )

    assert not recs.empty
    assert recs.iloc[0]["contribution_type"] == "Collaborative Only"


def test_hybrid_custom_preferences(hybrid_setup):
    agent, _, _ = hybrid_setup

    recs = agent.recommend_for_custom_preferences(
        favorite_movie_ids=[1],
        preferred_genres=["Animation"],
        content_weight=0.5,
        collab_weight=0.5,
        top_n=2,
    )

    assert not recs.empty
    assert 1 not in recs["movieId"].values
    assert recs.iloc[0]["movieId"] == 4  # Toy Story 2
