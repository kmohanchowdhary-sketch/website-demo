"""Collaborative Filtering Recommendation Engine.

Implements Item-Based Collaborative Filtering using user-item rating patterns,
cosine / adjusted cosine similarities, rating prediction formulas,
and explainability mechanisms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    """Item-Based Collaborative Filtering Recommender Engine."""

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        min_movie_ratings: int = 3,
        k_neighbors: int = 20,
    ):
        """Initialize the Collaborative Filtering Recommender.

        Args:
            ratings_df: DataFrame with 'userId', 'movieId', 'rating'.
            movies_df: DataFrame with 'movieId', 'title', 'genres', 'year'.
            min_movie_ratings: Minimum ratings required for a movie to be
              included in CF matrix to reduce noise.
            k_neighbors: Number of nearest item neighbors to consider during
              prediction.
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.min_movie_ratings = min_movie_ratings
        self.k_neighbors = k_neighbors

        # Metadata lookup maps
        self.movie_meta = self.movies_df.set_index("movieId").to_dict(
            orient="index"
        )

        # Global rating statistics
        self.global_mean = float(self.ratings_df["rating"].mean())

        # Build user-item and item-user representations
        self._build_matrices()

    def _build_matrices(self) -> None:
        """Construct user-item rating matrices and compute item similarities."""
        # Filter movies with sufficient ratings to ensure reliable item similarity
        movie_counts = self.ratings_df["movieId"].value_counts()
        valid_movie_ids = set(
            movie_counts[movie_counts >= self.min_movie_ratings].index
        )

        filtered_ratings = self.ratings_df[
            self.ratings_df["movieId"].isin(valid_movie_ids)
        ].copy()

        # Unique IDs
        self.user_ids = np.sort(filtered_ratings["userId"].unique())
        self.movie_ids = np.sort(filtered_ratings["movieId"].unique())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(self.user_ids)}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(self.movie_ids)}

        num_users = len(self.user_ids)
        num_movies = len(self.movie_ids)

        # Map to indices
        user_indices = filtered_ratings["userId"].map(self.user_to_idx).to_numpy()
        movie_indices = filtered_ratings["movieId"].map(self.movie_to_idx).to_numpy()
        ratings_values = filtered_ratings["rating"].to_numpy()

        # Mean rating per user for centered cosine (Pearson correlation equivalent)
        user_mean_series = filtered_ratings.groupby("userId")["rating"].mean()
        self.user_means = {
            uid: float(user_mean_series.get(uid, self.global_mean))
            for uid in self.user_ids
        }

        # Item mean ratings
        movie_mean_series = filtered_ratings.groupby("movieId")["rating"].mean()
        self.movie_means = {
            mid: float(movie_mean_series.get(mid, self.global_mean))
            for mid in self.movie_ids
        }

        # User-Item Sparse Matrix (rows: users, cols: movies)
        self.user_item_matrix = csr_matrix(
            (ratings_values, (user_indices, movie_indices)),
            shape=(num_users, num_movies),
            dtype=np.float32,
        )

        # Item-User Sparse Matrix (rows: movies, cols: users)
        # Centered by user mean rating for Adjusted Cosine Similarity
        centered_ratings = np.zeros_like(ratings_values, dtype=np.float32)
        for i, (u_idx, r) in enumerate(zip(user_indices, ratings_values)):
            uid = self.idx_to_user[u_idx]
            centered_ratings[i] = r - self.user_means[uid]

        self.item_user_centered = csr_matrix(
            (centered_ratings, (movie_indices, user_indices)),
            shape=(num_movies, num_users),
            dtype=np.float32,
        )

        # Precompute item-item cosine similarity matrix
        # Cosine similarity on user-centered vectors equals Adjusted Cosine (Pearson)
        print("Computing Item-Item similarity matrix for collaborative filtering...")
        self.item_similarity_matrix = cosine_similarity(
            self.item_user_centered, dense_output=True
        )
        # Set self-similarity to 0 to avoid recommending the same movie
        np.fill_diagonal(self.item_similarity_matrix, 0.0)

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find movies most similar to the given movie based on user rating patterns.

        Args:
            movie_id: Target movie ID.
            top_n: Number of similar movies to return.

        Returns:
            List of (movieId, similarity_score) tuples.
        """
        if movie_id not in self.movie_to_idx:
            return []

        item_idx = self.movie_to_idx[movie_id]
        sims = self.item_similarity_matrix[item_idx]

        sorted_indices = np.argsort(sims)[::-1]
        results = []
        for idx in sorted_indices:
            score = float(sims[idx])
            if score <= 0.001:
                break
            cand_id = self.idx_to_movie[idx]
            results.append((cand_id, score))
            if len(results) >= top_n:
                break
        return results

    def predict_rating(
        self, user_id: int, movie_id: int, user_ratings_dict: Optional[Dict[int, float]] = None
    ) -> Tuple[float, List[Tuple[int, float, float]]]:
        """Predict user's rating for a specific movie using Item-Based CF.

        Formula:
            r_hat(u, i) = sum(sim(i, j) * r(u, j)) / sum(|sim(i, j)|)
            over top-K rated neighbors j.

        Args:
            user_id: User ID.
            movie_id: Target movie ID to predict.
            user_ratings_dict: Optional mapping of movieId -> rating for the user
              (if omitted, read from internal matrix).

        Returns:
            Tuple of (predicted_rating, contributing_items [(movieId, similarity, user_rating)])
        """
        if movie_id not in self.movie_to_idx:
            # Fallback to movie mean or global mean
            return self.movie_means.get(movie_id, self.global_mean), []

        target_idx = self.movie_to_idx[movie_id]
        sim_row = self.item_similarity_matrix[target_idx]

        # Get user's ratings
        if user_ratings_dict is None:
            if user_id not in self.user_to_idx:
                return self.movie_means.get(movie_id, self.global_mean), []
            u_idx = self.user_to_idx[user_id]
            user_ratings_row = self.user_item_matrix.getrow(u_idx).toarray().flatten()
            rated_indices = np.where(user_ratings_row > 0)[0]
            rated_items = {
                self.idx_to_movie[idx]: float(user_ratings_row[idx])
                for idx in rated_indices
            }
        else:
            rated_items = user_ratings_dict

        if not rated_items:
            return self.movie_means.get(movie_id, self.global_mean), []

        # Find similarities with movies the user has rated
        neighbor_similarities = []
        for rated_mid, r_val in rated_items.items():
            if rated_mid in self.movie_to_idx:
                r_idx = self.movie_to_idx[rated_mid]
                sim = float(sim_row[r_idx])
                if sim > 0:  # Only consider positive correlations
                    neighbor_similarities.append((rated_mid, sim, r_val))

        if not neighbor_similarities:
            return self.movie_means.get(movie_id, self.global_mean), []

        # Sort by similarity and keep top K
        neighbor_similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = neighbor_similarities[: self.k_neighbors]

        sim_sum = sum(sim for _, sim, _ in top_neighbors)
        if sim_sum == 0:
            return self.movie_means.get(movie_id, self.global_mean), []

        weighted_rating_sum = sum(sim * r for _, sim, r in top_neighbors)
        predicted = weighted_rating_sum / sim_sum

        # Clip predicted rating to valid bounds [0.5, 5.0]
        predicted = float(np.clip(predicted, 0.5, 5.0))
        return predicted, top_neighbors

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        user_ratings_dict: Optional[Dict[int, float]] = None,
        candidate_movie_ids: Optional[Set[int]] = None,
    ) -> pd.DataFrame:
        """Generate Collaborative Filtering recommendations for a user.

        Ranks unrated movies by their predicted rating score based on the user's
        ratings and item similarities.

        Args:
            user_id: User ID.
            top_n: Number of recommendations.
            user_ratings_dict: Optional dictionary of {movieId: rating}.
            candidate_movie_ids: Optional set of movie IDs to evaluate.

        Returns:
            DataFrame with recommendations, predicted scores, and explanations.
        """
        # Retrieve user ratings
        if user_ratings_dict is None:
            if user_id in self.user_to_idx:
                u_idx = self.user_to_idx[user_id]
                u_row = self.user_item_matrix.getrow(u_idx).toarray().flatten()
                rated_indices = np.where(u_row > 0)[0]
                user_ratings_dict = {
                    self.idx_to_movie[idx]: float(u_row[idx])
                    for idx in rated_indices
                }
            else:
                user_ratings_dict = {}

        rated_mids = set(user_ratings_dict.keys())

        # Cold start check
        if len(rated_mids) == 0:
            return self._handle_cold_start(top_n=top_n)

        # Vectorized batch prediction for all items
        r_vec = np.zeros(len(self.movie_ids), dtype=np.float32)
        mask_vec = np.zeros(len(self.movie_ids), dtype=np.float32)

        for mid, r_val in user_ratings_dict.items():
            if mid in self.movie_to_idx:
                idx = self.movie_to_idx[mid]
                r_vec[idx] = float(r_val)
                mask_vec[idx] = 1.0

        pos_sim = np.maximum(self.item_similarity_matrix, 0.0)
        weighted_sums = pos_sim @ r_vec
        sim_sums = pos_sim @ mask_vec

        pred_ratings = np.zeros(len(self.movie_ids), dtype=np.float32)
        valid_mask = sim_sums > 1e-6
        pred_ratings[valid_mask] = np.clip(
            weighted_sums[valid_mask] / sim_sums[valid_mask], 0.5, 5.0
        )

        # Candidate filtering
        allowed_indices = set(range(len(self.movie_ids)))
        if candidate_movie_ids is not None:
            allowed_indices = {
                self.movie_to_idx[mid]
                for mid in candidate_movie_ids
                if mid in self.movie_to_idx
            }

        # Exclude already rated movies and unallowed candidates
        for mid in rated_mids:
            if mid in self.movie_to_idx:
                pred_ratings[self.movie_to_idx[mid]] = -1.0

        for idx in range(len(self.movie_ids)):
            if idx not in allowed_indices:
                pred_ratings[idx] = -1.0

        # Sort indices by predicted rating
        sorted_indices = np.argsort(pred_ratings)[::-1]

        scored_candidates = []
        for idx in sorted_indices:
            score = float(pred_ratings[idx])
            if score <= 0:
                break
            mid = self.idx_to_movie[idx]

            # Find top contributing rated item for explainability
            meta = self.movie_meta.get(mid, {})
            sim_row = pos_sim[idx]
            best_contributor_mid = None
            best_sim = 0.0
            best_r = 0.0

            for rated_mid, r_val in user_ratings_dict.items():
                if rated_mid in self.movie_to_idx:
                    r_idx = self.movie_to_idx[rated_mid]
                    s_val = float(sim_row[r_idx])
                    if s_val > best_sim:
                        best_sim = s_val
                        best_contributor_mid = rated_mid
                        best_r = r_val

            if best_contributor_mid and best_sim > 0.01:
                best_contributor_title = self.movie_meta.get(
                    best_contributor_mid, {}
                ).get("title", f"Movie #{best_contributor_mid}")
                explanation = (
                    f"Correlated with '{best_contributor_title}' (similarity {round(best_sim, 2)}) "
                    f"which you rated {best_r}★."
                )
            else:
                explanation = "Based on aggregate community rating trends for similar movies."

            scored_candidates.append(
                {
                    "movieId": mid,
                    "title": meta.get("title", f"Movie #{mid}"),
                    "clean_title": meta.get("clean_title", f"Movie #{mid}"),
                    "genres": meta.get("genres", ""),
                    "year": meta.get("year", None),
                    "predicted_rating": round(score, 2),
                    "collab_score": round(score / 5.0, 4),
                    "explanation": explanation,
                }
            )

            if len(scored_candidates) >= top_n:
                break

        if not scored_candidates:
            return self._handle_cold_start(top_n=top_n, exclude=rated_mids)

        df_results = pd.DataFrame(scored_candidates)
        return df_results.reset_index(drop=True)

    def _handle_cold_start(
        self, top_n: int = 10, exclude: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """Fallback for users with no collaborative rating history.

        Returns highest-rated movies with substantial ratings.
        """
        exclude = exclude or set()
        candidates = []
        for mid, mean_r in self.movie_means.items():
            if mid in exclude:
                continue
            meta = self.movie_meta.get(mid, {})
            candidates.append(
                {
                    "movieId": mid,
                    "title": meta.get("title", f"Movie #{mid}"),
                    "clean_title": meta.get("clean_title", f"Movie #{mid}"),
                    "genres": meta.get("genres", ""),
                    "year": meta.get("year", None),
                    "predicted_rating": round(mean_r, 2),
                    "collab_score": round(mean_r / 5.0, 4),
                    "explanation": "Cold-start fallback: Top rated by the MovieLens community.",
                }
            )
        df = pd.DataFrame(candidates)
        if df.empty:
            return pd.DataFrame()
        return df.sort_values(by="predicted_rating", ascending=False).head(top_n).reset_index(drop=True)
