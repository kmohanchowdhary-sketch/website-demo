"""Hybrid Recommendation Agent combining Content-Based and Collaborative Filtering.

Normalizes and fuses scores from both recommendation engines, adapts dynamically
to user-specified weights, handles cold-start scenarios, and provides detailed
explanations of model contributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.collaborative import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender


class HybridRecommendationAgent:
    """Intelligent Hybrid Recommendation Agent."""

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collab_recommender: CollaborativeFilteringRecommender,
    ):
        """Initialize the Hybrid Agent with Content-Based and Collaborative engines."""
        self.cb = content_recommender
        self.cf = collab_recommender
        self.movies_df = self.cb.movies_df
        self.movie_meta = self.movies_df.set_index("movieId").to_dict(
            orient="index"
        )

    def validate_and_normalize_weights(
        self, content_weight: float, collab_weight: float
    ) -> Tuple[float, float]:
        """Validate and normalize recommendation weights so they sum to 1.0.

        Args:
            content_weight: Raw content-based weight (>= 0).
            collab_weight: Raw collaborative filtering weight (>= 0).

        Returns:
            Tuple of (normalized_content_weight, normalized_collab_weight).

        Raises:
            ValueError: If weights are negative or both sum to 0.
        """
        if content_weight < 0 or collab_weight < 0:
            raise ValueError("Weights cannot be negative.")

        total = content_weight + collab_weight
        if total <= 1e-9:
            # Default to equal balance if both are 0
            return 0.5, 0.5

        return round(content_weight / total, 4), round(collab_weight / total, 4)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max scale an array of scores into [0.0, 1.0]."""
        if len(scores) == 0:
            return scores
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.ones_like(scores)

    def recommend_for_user(
        self,
        user_id: int,
        user_ratings_df: pd.DataFrame,
        content_weight: float = 0.5,
        collab_weight: float = 0.5,
        top_n: int = 10,
        candidate_pool_size: int = 100,
        candidate_movie_ids: Optional[Set[int]] = None,
    ) -> pd.DataFrame:
        """Generate hybrid personalized recommendations for a dataset user.

        Args:
            user_id: User ID.
            user_ratings_df: DataFrame of user's ratings ('movieId', 'rating').
            content_weight: Weight given to Content-Based scores.
            collab_weight: Weight given to Collaborative Filtering scores.
            top_n: Number of recommendations to return.
            candidate_pool_size: Number of top candidate items fetched from each model before merging.
            candidate_movie_ids: Optional set of movie IDs to restrict candidate generation to.

        Returns:
            DataFrame containing ranked recommendations, score breakdown,
            contribution types, and explanations.
        """
        w_cb, w_cf = self.validate_and_normalize_weights(
            content_weight, collab_weight
        )

        rated_mids: Set[int] = set()
        if not user_ratings_df.empty:
            rated_mids = set(user_ratings_df["movieId"].tolist())

        user_ratings_dict = (
            user_ratings_df.set_index("movieId")["rating"].to_dict()
            if not user_ratings_df.empty
            else {}
        )

        # 1. Edge Case: Pure Content-Based (w_cf == 0)
        if w_cf <= 1e-4:
            cb_df = self.cb.recommend_for_user(
                user_id=user_id,
                user_ratings_df=user_ratings_df,
                top_n=top_n,
                candidate_movie_ids=candidate_movie_ids,
            )
            if cb_df.empty:
                return pd.DataFrame()
            cb_df["hybrid_score"] = cb_df["content_score"]
            cb_df["collab_score"] = 0.0
            cb_df["contribution_type"] = "Content-Based Only"
            return cb_df.head(top_n)

        # 2. Edge Case: Pure Collaborative Filtering (w_cb == 0)
        if w_cb <= 1e-4:
            cf_df = self.cf.recommend_for_user(
                user_id=user_id,
                top_n=top_n,
                user_ratings_dict=user_ratings_dict,
                candidate_movie_ids=candidate_movie_ids,
            )
            if cf_df.empty:
                return pd.DataFrame()
            cf_df["hybrid_score"] = cf_df["collab_score"]
            cf_df["content_score"] = 0.0
            cf_df["contribution_type"] = "Collaborative Only"
            return cf_df.head(top_n)

        # 3. Hybrid Combination: Retrieve Candidate Pools from both algorithms
        cb_candidates = self.cb.recommend_for_user(
            user_id=user_id,
            user_ratings_df=user_ratings_df,
            top_n=candidate_pool_size,
            candidate_movie_ids=candidate_movie_ids,
        )

        cf_candidates = self.cf.recommend_for_user(
            user_id=user_id,
            top_n=candidate_pool_size,
            user_ratings_dict=user_ratings_dict,
            candidate_movie_ids=candidate_movie_ids,
        )

        # Cold-Start Check: If CF has no data, fall back gracefully to Content-Based
        if cf_candidates.empty and not cb_candidates.empty:
            cb_candidates["hybrid_score"] = cb_candidates["content_score"]
            cb_candidates["collab_score"] = 0.0
            cb_candidates["contribution_type"] = "Content-Based (Cold-Start Fallback)"
            return cb_candidates.head(top_n)

        # If CB has no data, fall back to CF
        if cb_candidates.empty and not cf_candidates.empty:
            cf_candidates["hybrid_score"] = cf_candidates["collab_score"]
            cf_candidates["content_score"] = 0.0
            cf_candidates["contribution_type"] = "Collaborative (Cold-Start Fallback)"
            return cf_candidates.head(top_n)

        if cb_candidates.empty and cf_candidates.empty:
            return pd.DataFrame()

        # Build combined candidate map
        candidate_data: Dict[int, Dict[str, Any]] = {}

        # Process Content-Based Candidates
        cb_scores_arr = cb_candidates["content_score"].to_numpy()
        cb_norm_scores = self._normalize_scores(cb_scores_arr)

        for i, row in cb_candidates.iterrows():
            mid = int(row["movieId"])
            if mid in rated_mids:
                continue
            candidate_data[mid] = {
                "movieId": mid,
                "title": row["title"],
                "clean_title": row["clean_title"],
                "genres": row["genres"],
                "year": row["year"],
                "raw_content_score": float(row["content_score"]),
                "content_score": float(cb_norm_scores[i]),
                "raw_collab_score": 0.0,
                "collab_score": 0.0,
                "cb_explanation": row["explanation"],
                "cf_explanation": "",
                "has_cb": True,
                "has_cf": False,
            }

        # Process Collaborative Candidates
        cf_scores_arr = cf_candidates["collab_score"].to_numpy()
        cf_norm_scores = self._normalize_scores(cf_scores_arr)

        for i, row in cf_candidates.iterrows():
            mid = int(row["movieId"])
            if mid in rated_mids:
                continue
            if mid in candidate_data:
                # Item was identified by BOTH algorithms!
                candidate_data[mid]["raw_collab_score"] = float(row["collab_score"])
                candidate_data[mid]["collab_score"] = float(cf_norm_scores[i])
                candidate_data[mid]["cf_explanation"] = row["explanation"]
                candidate_data[mid]["has_cf"] = True
            else:
                candidate_data[mid] = {
                    "movieId": mid,
                    "title": row["title"],
                    "clean_title": row["clean_title"],
                    "genres": row["genres"],
                    "year": row["year"],
                    "raw_content_score": 0.0,
                    "content_score": 0.0,
                    "raw_collab_score": float(row["collab_score"]),
                    "collab_score": float(cf_norm_scores[i]),
                    "cb_explanation": "",
                    "cf_explanation": row["explanation"],
                    "has_cb": False,
                    "has_cf": True,
                }

        # Calculate hybrid score and synthesized explanation for each candidate
        results = []
        for mid, data in candidate_data.items():
            s_cb = data["content_score"]
            s_cf = data["collab_score"]
            has_cb = data["has_cb"]
            has_cf = data["has_cf"]

            if has_cb and has_cf:
                # Genuine Hybrid: both algorithms support this recommendation
                hybrid_score = (w_cb * s_cb) + (w_cf * s_cf)
                contrib = "Hybrid (Both Models)"
                explanation = (
                    f"Recommended by both models. {data['cb_explanation']} "
                    f"Also supported by collaborative patterns: {data['cf_explanation']}"
                )
            elif has_cb:
                # Only Content-Based provided this item
                # Penalize or weight by content weight
                hybrid_score = w_cb * s_cb
                contrib = "Content-Based Only"
                explanation = data["cb_explanation"]
            else:
                # Only Collaborative Filtering provided this item
                hybrid_score = w_cf * s_cf
                contrib = "Collaborative Only"
                explanation = data["cf_explanation"]

            results.append(
                {
                    "movieId": mid,
                    "title": data["title"],
                    "clean_title": data["clean_title"],
                    "genres": data["genres"],
                    "year": data["year"],
                    "hybrid_score": round(float(hybrid_score), 4),
                    "content_score": round(float(s_cb), 4),
                    "collab_score": round(float(s_cf), 4),
                    "contribution_type": contrib,
                    "explanation": explanation,
                }
            )

        df_results = pd.DataFrame(results)
        if df_results.empty:
            return pd.DataFrame()

        df_results = df_results.sort_values(
            by=["hybrid_score", "movieId"], ascending=[False, True]
        ).head(top_n)

        return df_results.reset_index(drop=True)

    def recommend_for_custom_preferences(
        self,
        favorite_movie_ids: List[int],
        preferred_genres: List[str],
        content_weight: float = 0.5,
        collab_weight: float = 0.5,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Generate hybrid recommendations for a cold-start user based on explicit preferences."""
        w_cb, w_cf = self.validate_and_normalize_weights(
            content_weight, collab_weight
        )

        # Create simulated ratings (5.0 stars for favorite movies)
        simulated_ratings = {mid: 5.0 for mid in favorite_movie_ids}
        simulated_df = pd.DataFrame(
            [{"movieId": mid, "rating": 5.0} for mid in favorite_movie_ids]
        )

        # 1. Content-based candidates based on favorites + preferred genres
        cb_candidates = self.cb.recommend_for_custom_preferences(
            favorite_movie_ids=favorite_movie_ids,
            preferred_genres=preferred_genres,
            top_n=50,
            excluded_movie_ids=set(favorite_movie_ids),
        )

        # 2. Collaborative candidates based on simulated ratings
        cf_candidates = self.cf.recommend_for_user(
            user_id=-1,  # Temporary cold-start ID
            top_n=50,
            user_ratings_dict=simulated_ratings,
        )

        # If pure content or CF is empty
        if cf_candidates.empty or w_cf <= 1e-4:
            if cb_candidates.empty:
                return pd.DataFrame()
            cb_candidates["hybrid_score"] = cb_candidates["content_score"]
            cb_candidates["collab_score"] = 0.0
            cb_candidates["contribution_type"] = "Content-Based Only"
            return cb_candidates.head(top_n)

        if cb_candidates.empty or w_cb <= 1e-4:
            cf_candidates["hybrid_score"] = cf_candidates["collab_score"]
            cf_candidates["content_score"] = 0.0
            cf_candidates["contribution_type"] = "Collaborative Only"
            return cf_candidates.head(top_n)

        # Merge candidates
        cb_norm = self._normalize_scores(cb_candidates["content_score"].to_numpy())
        cf_norm = self._normalize_scores(cf_candidates["collab_score"].to_numpy())

        candidate_data: Dict[int, Dict[str, Any]] = {}
        for i, row in cb_candidates.iterrows():
            mid = int(row["movieId"])
            candidate_data[mid] = {
                "movieId": mid,
                "title": row["title"],
                "clean_title": row["clean_title"],
                "genres": row["genres"],
                "year": row["year"],
                "content_score": float(cb_norm[i]),
                "collab_score": 0.0,
                "cb_explanation": row["explanation"],
                "cf_explanation": "",
                "has_cb": True,
                "has_cf": False,
            }

        for i, row in cf_candidates.iterrows():
            mid = int(row["movieId"])
            if mid in candidate_data:
                candidate_data[mid]["collab_score"] = float(cf_norm[i])
                candidate_data[mid]["cf_explanation"] = row["explanation"]
                candidate_data[mid]["has_cf"] = True
            else:
                candidate_data[mid] = {
                    "movieId": mid,
                    "title": row["title"],
                    "clean_title": row["clean_title"],
                    "genres": row["genres"],
                    "year": row["year"],
                    "content_score": 0.0,
                    "collab_score": float(cf_norm[i]),
                    "cb_explanation": "",
                    "cf_explanation": row["explanation"],
                    "has_cb": False,
                    "has_cf": True,
                }

        results = []
        for mid, data in candidate_data.items():
            s_cb = data["content_score"]
            s_cf = data["collab_score"]
            if data["has_cb"] and data["has_cf"]:
                hybrid_score = (w_cb * s_cb) + (w_cf * s_cf)
                contrib = "Hybrid (Both Models)"
                explanation = (
                    f"Matches your chosen genres/favorites. {data['cb_explanation']} "
                    f"Additionally, users who liked your favorites also recommended this: {data['cf_explanation']}"
                )
            elif data["has_cb"]:
                hybrid_score = w_cb * s_cb
                contrib = "Content-Based Only"
                explanation = data["cb_explanation"]
            else:
                hybrid_score = w_cf * s_cf
                contrib = "Collaborative Only"
                explanation = data["cf_explanation"]

            results.append(
                {
                    "movieId": mid,
                    "title": data["title"],
                    "clean_title": data["clean_title"],
                    "genres": data["genres"],
                    "year": data["year"],
                    "hybrid_score": round(float(hybrid_score), 4),
                    "content_score": round(float(s_cb), 4),
                    "collab_score": round(float(s_cf), 4),
                    "contribution_type": contrib,
                    "explanation": explanation,
                }
            )

        df = pd.DataFrame(results).sort_values(
            by=["hybrid_score", "movieId"], ascending=[False, True]
        ).head(top_n)

        return df.reset_index(drop=True)
