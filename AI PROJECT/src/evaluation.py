"""Offline Evaluation Framework for Recommendation Algorithms.

Implements train/test splitting without data leakage, ranking evaluation metrics
(Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR@K), rating error metrics
(RMSE, MAE), and cross-algorithm benchmarking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class RecommenderEvaluator:
    """Evaluates recommendation models using standard offline metrics."""

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        test_ratio: float = 0.2,
        min_user_ratings: int = 15,
        relevance_threshold: float = 3.5,
        random_state: int = 42,
    ):
        """Initialize the evaluator and split data.

        Args:
            ratings_df: Full ratings DataFrame ('userId', 'movieId', 'rating',
              'timestamp').
            movies_df: Movies catalog DataFrame.
            test_ratio: Proportion of ratings held out for testing per user.
            min_user_ratings: Minimum ratings required for a user to qualify for
              evaluation.
            relevance_threshold: Minimum rating to consider a movie 'relevant' /
              positive.
            random_state: Random seed for reproducible splitting.
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.test_ratio = test_ratio
        self.min_user_ratings = min_user_ratings
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state

        self.train_ratings, self.test_ratings = self._split_train_test()

    def _split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform a stratified per-user train/test split.

        Holds out a proportion of each qualifying user's ratings for testing.
        Guarantees that test ratings are never seen during model training.
        """
        user_counts = self.ratings_df["userId"].value_counts()
        qualifying_users = set(
            user_counts[user_counts >= self.min_user_ratings].index
        )

        train_dfs = []
        test_dfs = []

        np.random.seed(self.random_state)

        for uid, group in self.ratings_df.groupby("userId"):
            if uid in qualifying_users and len(group) >= self.min_user_ratings:
                # Shuffle group
                shuffled = group.sample(
                    frac=1.0, random_state=self.random_state
                )
                n_test = max(1, int(len(shuffled) * self.test_ratio))
                test_part = shuffled.iloc[:n_test]
                train_part = shuffled.iloc[n_test:]
                test_dfs.append(test_part)
                train_dfs.append(train_part)
            else:
                train_dfs.append(group)

        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = (
            pd.concat(test_dfs, ignore_index=True)
            if test_dfs
            else pd.DataFrame(columns=self.ratings_df.columns)
        )

        return train_df, test_df

    @staticmethod
    def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """Calculate Precision@K: fraction of top-K recommendations that are relevant."""
        if k <= 0:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for mid in top_k if mid in relevant_ids)
        return float(hits / k)

    @staticmethod
    def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """Calculate Recall@K: fraction of relevant items captured in top-K."""
        if not relevant_ids:
            return 0.0
        top_k = recommended_ids[:k]
        hits = sum(1 for mid in top_k if mid in relevant_ids)
        return float(hits / len(relevant_ids))

    @staticmethod
    def hit_rate_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """Calculate Hit Rate@K: 1.0 if at least one relevant item is in top-K, else 0.0."""
        top_k = recommended_ids[:k]
        return 1.0 if any(mid in relevant_ids for mid in top_k) else 0.0

    @staticmethod
    def mrr_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """Calculate Mean Reciprocal Rank at K (MRR@K)."""
        top_k = recommended_ids[:k]
        for rank, mid in enumerate(top_k, start=1):
            if mid in relevant_ids:
                return float(1.0 / rank)
        return 0.0

    @staticmethod
    def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance."""
        if not relevant_ids or k <= 0:
            return 0.0

        top_k = recommended_ids[:k]
        # DCG@K
        dcg = 0.0
        for i, mid in enumerate(top_k):
            if mid in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because 0-indexed: i=0 -> rank 1 -> log2(2) = 1.0

        # Ideal DCG@K: top items are all relevant up to min(k, len(relevant_ids))
        idcg = sum(
            1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids)))
        )
        if idcg <= 0.0:
            return 0.0

        return float(dcg / idcg)

    @staticmethod
    def calculate_rmse_mae(
        actual_ratings: List[float], predicted_ratings: List[float]
    ) -> Tuple[float, float]:
        """Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
        if not actual_ratings or len(actual_ratings) != len(predicted_ratings):
            return 0.0, 0.0

        y_true = np.array(actual_ratings)
        y_pred = np.array(predicted_ratings)

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return round(rmse, 4), round(mae, 4)

    def evaluate_models(
        self,
        content_recommender: Any,
        collab_recommender: Any,
        hybrid_agent: Any,
        k_values: List[int] = [5, 10, 20],
        max_eval_users: int = 40,
        content_weight: float = 0.5,
        collab_weight: float = 0.5,
        eval_mode: str = "sampled",
    ) -> Dict[str, Any]:
        """Run benchmark evaluation comparing Content-Based, Collaborative, and Hybrid models.

        Args:
            content_recommender: ContentBasedRecommender instance.
            collab_recommender: CollaborativeFilteringRecommender instance (trained on train_ratings).
            hybrid_agent: HybridRecommendationAgent instance.
            k_values: Cutoff K values (e.g. 5, 10, 20).
            max_eval_users: Number of test users to evaluate for execution speed.
            content_weight: Content weight for hybrid agent.
            collab_weight: Collaborative weight for hybrid agent.
            eval_mode: "sampled" (ranks relevant test items among 100 sampled negatives,
                       standard academic protocol) or "full" (ranks across entire catalog).

        Returns:
            Dictionary with evaluation metrics per model and summary dataframes.
        """
        # Find test users with at least 2 relevant items in test set
        test_user_groups = self.test_ratings.groupby("userId")
        candidate_uids = []
        for uid, grp in test_user_groups:
            rel_items = grp[grp["rating"] >= self.relevance_threshold]
            if len(rel_items) >= 2:
                candidate_uids.append(uid)

        # Subsample users for fast evaluation
        np.random.seed(self.random_state)
        eval_users = (
            np.random.choice(
                candidate_uids,
                size=min(len(candidate_uids), max_eval_users),
                replace=False,
            )
            if candidate_uids
            else []
        )

        models = ["Content-Based", "Collaborative", "Hybrid"]
        results: Dict[str, Dict[str, List[float]]] = {
            m: {f"P@{k}": [] for k in k_values} for m in models
        }
        for m in models:
            for k in k_values:
                results[m][f"R@{k}"] = []
                results[m][f"NDCG@{k}"] = []
                results[m][f"HitRate@{k}"] = []
            results[m]["MRR@10"] = []

        max_k = max(k_values)
        all_catalog_mids = set(self.movies_df["movieId"].tolist())

        for uid in eval_users:
            user_test_df = self.test_ratings[self.test_ratings["userId"] == uid]
            relevant_items = set(
                user_test_df[user_test_df["rating"] >= self.relevance_threshold]["movieId"]
            )

            if not relevant_items:
                continue

            user_train_df = self.train_ratings[self.train_ratings["userId"] == uid]
            user_train_dict = user_train_df.set_index("movieId")["rating"].to_dict()

            # Construct candidate pool according to evaluation protocol
            candidate_ids: Optional[Set[int]] = None
            if eval_mode == "sampled":
                user_all_rated = set(self.ratings_df[self.ratings_df["userId"] == uid]["movieId"])
                unrated_pool = list(all_catalog_mids - user_all_rated)
                if unrated_pool:
                    sample_size = min(len(unrated_pool), 100)
                    sampled_negatives = np.random.choice(unrated_pool, size=sample_size, replace=False)
                    candidate_ids = relevant_items | set(sampled_negatives)

            # 1. Content-Based recommendations
            cb_recs = content_recommender.recommend_for_user(
                user_id=uid,
                user_ratings_df=user_train_df,
                top_n=max_k,
                candidate_movie_ids=candidate_ids,
            )
            cb_ids = cb_recs["movieId"].tolist() if not cb_recs.empty else []

            # 2. Collaborative recommendations
            cf_recs = collab_recommender.recommend_for_user(
                user_id=uid,
                top_n=max_k,
                user_ratings_dict=user_train_dict,
                candidate_movie_ids=candidate_ids,
            )
            cf_ids = cf_recs["movieId"].tolist() if not cf_recs.empty else []

            # 3. Hybrid recommendations
            hyb_recs = hybrid_agent.recommend_for_user(
                user_id=uid,
                user_ratings_df=user_train_df,
                content_weight=content_weight,
                collab_weight=collab_weight,
                top_n=max_k,
                candidate_pool_size=len(candidate_ids) if candidate_ids else 100,
                candidate_movie_ids=candidate_ids,
            )
            hyb_ids = hyb_recs["movieId"].tolist() if not hyb_recs.empty else []

            rec_map = {
                "Content-Based": cb_ids,
                "Collaborative": cf_ids,
                "Hybrid": hyb_ids,
            }

            for model_name, rec_list in rec_map.items():
                for k in k_values:
                    p = self.precision_at_k(rec_list, relevant_items, k)
                    r = self.recall_at_k(rec_list, relevant_items, k)
                    ndcg = self.ndcg_at_k(rec_list, relevant_items, k)
                    hr = self.hit_rate_at_k(rec_list, relevant_items, k)

                    results[model_name][f"P@{k}"].append(p)
                    results[model_name][f"R@{k}"].append(r)
                    results[model_name][f"NDCG@{k}"].append(ndcg)
                    results[model_name][f"HitRate@{k}"].append(hr)

                mrr = self.mrr_at_k(rec_list, relevant_items, 10)
                results[model_name]["MRR@10"].append(mrr)

        # Aggregate averages
        summary_rows = []
        for model_name in models:
            row = {"Algorithm": model_name}
            for metric_key, val_list in results[model_name].items():
                mean_val = float(np.mean(val_list)) if val_list else 0.0
                row[metric_key] = round(mean_val, 4)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Rating prediction RMSE / MAE for Collaborative Filtering
        actuals = []
        preds = []
        for _, test_row in (
            self.test_ratings.head(500).iterrows()
        ):  # Sample for speed
            uid = int(test_row["userId"])
            mid = int(test_row["movieId"])
            actual_r = float(test_row["rating"])

            # Check if user had training data
            user_train = self.train_ratings[self.train_ratings["userId"] == uid]
            if not user_train.empty:
                train_dict = user_train.set_index("movieId")["rating"].to_dict()
                pred_r, _ = collab_recommender.predict_rating(
                    user_id=uid, movie_id=mid, user_ratings_dict=train_dict
                )
                actuals.append(actual_r)
                preds.append(pred_r)

        rmse, mae = self.calculate_rmse_mae(actuals, preds)

        return {
            "summary_df": summary_df,
            "raw_results": results,
            "k_values": k_values,
            "num_evaluated_users": len(eval_users),
            "cf_rmse": rmse,
            "cf_mae": mae,
        }
