"""Unit tests for RecommenderEvaluator and metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import RecommenderEvaluator


@pytest.fixture
def sample_eval_ratings():
    rows = []
    # Create 3 users with 20 ratings each to pass min_user_ratings
    for uid in [1, 2, 3]:
        for mid in range(1, 21):
            r = 4.0 if mid % 2 == 0 else 2.5
            rows.append({"userId": uid, "movieId": mid, "rating": r, "timestamp": 1000})

    ratings_df = pd.DataFrame(rows)
    movies_df = pd.DataFrame(
        [{"movieId": mid, "title": f"Movie {mid}", "genres": "Action"} for mid in range(1, 21)]
    )
    return ratings_df, movies_df


def test_train_test_split_no_leakage(sample_eval_ratings):
    ratings_df, movies_df = sample_eval_ratings
    evaluator = RecommenderEvaluator(
        ratings_df=ratings_df,
        movies_df=movies_df,
        test_ratio=0.2,
        min_user_ratings=10,
    )

    train_df = evaluator.train_ratings
    test_df = evaluator.test_ratings

    assert not train_df.empty
    assert not test_df.empty

    # Check strictly disjoint (no leakage)
    train_pairs = set(zip(train_df["userId"], train_df["movieId"]))
    test_pairs = set(zip(test_df["userId"], test_df["movieId"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_precision_recall_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6, 8}

    # Top-5 contains 2 hits (2, 4) out of 5
    p5 = RecommenderEvaluator.precision_at_k(recommended, relevant, k=5)
    assert p5 == 2 / 5

    # Recall: 2 hits out of 4 relevant items
    r5 = RecommenderEvaluator.recall_at_k(recommended, relevant, k=5)
    assert r5 == 2 / 4

    # Top-2 contains 1 hit (2)
    p2 = RecommenderEvaluator.precision_at_k(recommended, relevant, k=2)
    assert p2 == 1 / 2

    # Zero cases
    assert RecommenderEvaluator.precision_at_k(recommended, set(), k=5) == 0.0
    assert RecommenderEvaluator.recall_at_k(recommended, set(), k=5) == 0.0


def test_hit_rate_and_mrr():
    recommended = [10, 20, 30]
    relevant = {20}

    hr = RecommenderEvaluator.hit_rate_at_k(recommended, relevant, k=3)
    assert hr == 1.0

    # 20 is at rank 2 (1-indexed) -> MRR = 1/2 = 0.5
    mrr = RecommenderEvaluator.mrr_at_k(recommended, relevant, k=3)
    assert mrr == 0.5


def test_ndcg_at_k():
    # Recommended has hit at rank 1
    rec1 = [1, 2, 3]
    rel1 = {1}
    ndcg1 = RecommenderEvaluator.ndcg_at_k(rec1, rel1, k=3)
    assert ndcg1 == 1.0  # Perfect hit at position 1

    # Recommended has hit at rank 2
    rec2 = [2, 1, 3]
    ndcg2 = RecommenderEvaluator.ndcg_at_k(rec2, rel1, k=3)
    assert 0 < ndcg2 < 1.0
    assert ndcg2 < ndcg1  # Rank 2 is discounted relative to rank 1


def test_rmse_mae():
    actual = [4.0, 3.0, 5.0, 2.0]
    predicted = [4.0, 3.5, 4.5, 2.5]

    rmse, mae = RecommenderEvaluator.calculate_rmse_mae(actual, predicted)
    assert rmse == 0.433
    assert mae == 0.375
