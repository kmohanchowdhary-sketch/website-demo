"""Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.

Builds feature vectors from movie genres and user tags, constructs user profile
vectors from historical preferences, and computes cosine similarities to generate
explainable recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Recommends movies using item feature similarity and user preference profiles."""

    def __init__(self, movies_df: pd.DataFrame):
        """Initialize the Content-Based Recommender.

        Args:
            movies_df: DataFrame containing at least 'movieId', 'title',
              'genres', and 'metadata_soup'.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        # Mapping helpers
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self.idx_to_movie_id = {
            idx: mid for idx, mid in enumerate(self.movies_df["movieId"])
        }

        # Initialize TF-IDF Vectorizer
        # We handle single-letter or hyphenated tokens like 'sci-fi'
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            ngram_range=(1, 2),
            min_df=1,
        )

        # Fit and transform metadata soup
        soup_series = self.movies_df["metadata_soup"].fillna("")
        self.tfidf_matrix = self.vectorizer.fit_transform(soup_series)

    def get_movie_vector(self, movie_id: int):
        """Retrieve the TF-IDF vector for a given movie."""
        if movie_id not in self.movie_id_to_idx:
            raise KeyError(f"Movie ID {movie_id} not found in catalog.")
        idx = self.movie_id_to_idx[movie_id]
        return self.tfidf_matrix[idx]

    def _explain_similarity(
        self, target_genres: str, candidate_genres: str, base_titles: List[str]
    ) -> str:
        """Generate human-readable explanation of why a movie is recommended."""
        target_set = {
            g.strip()
            for g in target_genres.split("|")
            if g.strip() and g.strip() != "(no genres listed)"
        }
        cand_set = {
            g.strip()
            for g in candidate_genres.split("|")
            if g.strip() and g.strip() != "(no genres listed)"
        }
        overlap = target_set.intersection(cand_set)

        if overlap:
            overlap_str = ", ".join(sorted(list(overlap)))
            if base_titles:
                base_ref = f"'{base_titles[0]}'" if len(base_titles) == 1 else f"'{base_titles[0]}' and others"
                return f"Shares key genre(s) [{overlap_str}] with your favorite {base_ref}."
            return f"Matches your preferred genre(s): [{overlap_str}]."
        return "Strong thematic and keyword metadata overlap with your preferences."

    def recommend_for_movie(
        self, movie_id: int, top_n: int = 10, excluded_movie_ids: Optional[Set[int]] = None
    ) -> pd.DataFrame:
        """Recommend movies most similar to a specific movie.

        Args:
            movie_id: Target movie ID to find similar items for.
            top_n: Number of recommendations to return.
            excluded_movie_ids: Movie IDs to exclude (e.g. already seen).

        Returns:
            DataFrame with recommendations, similarity scores, and explanations.
        """
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()

        target_idx = self.movie_id_to_idx[movie_id]
        target_vector = self.tfidf_matrix[target_idx]
        target_row = self.movies_df.iloc[target_idx]

        # Compute cosine similarity with all items
        sim_scores = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        exclude = {movie_id}
        if excluded_movie_ids:
            exclude.update(excluded_movie_ids)

        # Get sorted candidate indices
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        for idx in sorted_indices:
            candidate_id = self.idx_to_movie_id[idx]
            if candidate_id in exclude:
                continue

            score = float(sim_scores[idx])
            if score <= 0.001:
                break  # No similarity

            cand_row = self.movies_df.iloc[idx]
            explanation = self._explain_similarity(
                target_genres=target_row["genres"],
                candidate_genres=cand_row["genres"],
                base_titles=[target_row["title"]],
            )

            results.append(
                {
                    "movieId": candidate_id,
                    "title": cand_row["title"],
                    "clean_title": cand_row["clean_title"],
                    "genres": cand_row["genres"],
                    "year": cand_row["year"],
                    "content_score": round(score, 4),
                    "explanation": explanation,
                }
            )

            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def recommend_for_user(
        self,
        user_id: int,
        user_ratings_df: pd.DataFrame,
        top_n: int = 10,
        min_positive_rating: float = 3.0,
        candidate_movie_ids: Optional[Set[int]] = None,
    ) -> pd.DataFrame:
        """Generate content-based recommendations for a user based on their rating history.

        Constructs a user interest profile vector as a rating-weighted average of
        TF-IDF vectors for movies they enjoyed, then finds unrated movies closest to this profile.

        Args:
            user_id: Target user ID.
            user_ratings_df: DataFrame containing all ratings for this user
              ('movieId', 'rating').
            top_n: Number of recommendations.
            min_positive_rating: Minimum rating considered a positive preference.

        Returns:
            DataFrame with recommendations, scores, and explanations.
        """
        if user_ratings_df.empty:
            return pd.DataFrame()

        # Filter movies rated by user that exist in catalog
        valid_ratings = user_ratings_df[
            user_ratings_df["movieId"].isin(self.movie_id_to_idx)
        ].copy()

        if valid_ratings.empty:
            return pd.DataFrame()

        rated_movie_ids = set(valid_ratings["movieId"])

        # Focus on positively rated movies, or highest rated if none meet threshold
        positive_ratings = valid_ratings[
            valid_ratings["rating"] >= min_positive_rating
        ]
        if positive_ratings.empty:
            positive_ratings = valid_ratings.sort_values(
                by="rating", ascending=False
            ).head(5)

        # Build user profile vector: sum(weight_i * tfidf_i)
        # Weights normalized so higher ratings have proportionally stronger influence
        min_r = positive_ratings["rating"].min()
        max_r = positive_ratings["rating"].max()
        if max_r > min_r:
            weights = (positive_ratings["rating"] - min_r + 0.5) / (
                max_r - min_r + 0.5
            )
        else:
            weights = np.ones(len(positive_ratings))

        indices = [
            self.movie_id_to_idx[mid] for mid in positive_ratings["movieId"]
        ]
        sub_matrix = self.tfidf_matrix[indices]

        weight_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        weighted_vectors = sub_matrix.multiply(weight_arr)
        user_profile_vec = np.asarray(weighted_vectors.sum(axis=0))

        # Normalize profile vector
        norm = np.linalg.norm(user_profile_vec)
        if norm > 0:
            user_profile_vec = user_profile_vec / norm

        # Compute cosine similarity between user profile and all movies
        sim_scores = cosine_similarity(user_profile_vec, self.tfidf_matrix).flatten()

        # Identify user's top favorite movies for explanation context
        top_positive = positive_ratings[["movieId", "rating"]].sort_values(
            by="rating", ascending=False
        ).head(3)
        top_user_movies = top_positive.merge(
            self.movies_df[["movieId", "title", "genres"]], on="movieId", how="inner"
        )
        top_titles = top_user_movies["title"].tolist()
        top_genres_str = "|".join(top_user_movies["genres"].tolist())

        sorted_indices = np.argsort(sim_scores)[::-1]
        results = []

        for idx in sorted_indices:
            candidate_id = self.idx_to_movie_id[idx]
            if candidate_id in rated_movie_ids:
                continue
            if candidate_movie_ids is not None and candidate_id not in candidate_movie_ids:
                continue

            score = float(sim_scores[idx])
            cand_row = self.movies_df.iloc[idx]

            explanation = self._explain_similarity(
                target_genres=top_genres_str,
                candidate_genres=cand_row["genres"],
                base_titles=top_titles,
            )

            results.append(
                {
                    "movieId": candidate_id,
                    "title": cand_row["title"],
                    "clean_title": cand_row["clean_title"],
                    "genres": cand_row["genres"],
                    "year": cand_row["year"],
                    "content_score": round(score, 4),
                    "explanation": explanation,
                }
            )

            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def recommend_for_custom_preferences(
        self,
        favorite_movie_ids: List[int],
        preferred_genres: List[str],
        top_n: int = 10,
        excluded_movie_ids: Optional[Set[int]] = None,
    ) -> pd.DataFrame:
        """Generate content-based recommendations for cold-start users specifying favorites & genres."""
        exclude = set(favorite_movie_ids)
        if excluded_movie_ids:
            exclude.update(excluded_movie_ids)

        # Build composite text or vector
        vectors = []
        base_titles = []
        base_genres = []

        for mid in favorite_movie_ids:
            if mid in self.movie_id_to_idx:
                vectors.append(self.get_movie_vector(mid).toarray())
                row = self.movies_df.iloc[self.movie_id_to_idx[mid]]
                base_titles.append(row["title"])
                base_genres.append(row["genres"])

        if preferred_genres:
            genre_text = " ".join(preferred_genres).lower()
            genre_vec = self.vectorizer.transform([genre_text]).toarray()
            # Double weight for explicit genre picks
            vectors.append(genre_vec * 1.5)
            base_genres.extend(preferred_genres)

        if not vectors:
            return pd.DataFrame()

        # Combine into aggregate profile vector
        combined_vec = np.sum(vectors, axis=0)
        norm = np.linalg.norm(combined_vec)
        if norm > 0:
            combined_vec = combined_vec / norm

        sim_scores = cosine_similarity(combined_vec, self.tfidf_matrix).flatten()
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        target_genres_str = "|".join(base_genres)

        for idx in sorted_indices:
            candidate_id = self.idx_to_movie_id[idx]
            if candidate_id in exclude:
                continue

            score = float(sim_scores[idx])
            cand_row = self.movies_df.iloc[idx]

            explanation = self._explain_similarity(
                target_genres=target_genres_str,
                candidate_genres=cand_row["genres"],
                base_titles=base_titles,
            )

            results.append(
                {
                    "movieId": candidate_id,
                    "title": cand_row["title"],
                    "clean_title": cand_row["clean_title"],
                    "genres": cand_row["genres"],
                    "year": cand_row["year"],
                    "content_score": round(score, 4),
                    "explanation": explanation,
                }
            )

            if len(results) >= top_n:
                break

        return pd.DataFrame(results)
