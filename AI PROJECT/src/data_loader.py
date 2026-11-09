"""Data loader and preprocessing module for the MovieLens dataset.

Handles downloading, unzipping, caching, cleaning, metadata enrichment,
and querying of the MovieLens (ml-latest-small) dataset.
"""

from __future__ import annotations

import os
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MovieLensDataLoader:
    """Manages loading, cleaning, and statistics for the MovieLens dataset."""

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    DATA_FOLDER_NAME = "ml-latest-small"

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to directory where datasets are stored. Defaults to
              'data' in the current working directory.
        """
        if data_dir is None:
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)

        self.dataset_dir = self.data_dir / self.DATA_FOLDER_NAME
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.tags_df: Optional[pd.DataFrame] = None
        self.links_df: Optional[pd.DataFrame] = None

    def is_dataset_available(self) -> bool:
        """Check if required MovieLens CSV files exist locally."""
        movies_path = self.dataset_dir / "movies.csv"
        ratings_path = self.dataset_dir / "ratings.csv"
        return movies_path.is_file() and ratings_path.is_file()

    def download_and_extract(self, force: bool = False) -> Path:
        """Download and extract MovieLens dataset if not already present.

        Args:
            force: If True, re-downloads even if files exist.

        Returns:
            Path to the extracted dataset directory.

        Raises:
            RuntimeError: If download fails and files do not exist.
        """
        if not force and self.is_dataset_available():
            return self.dataset_dir

        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / "ml-latest-small.zip"

        try:
            print(f"Downloading MovieLens dataset from {self.DATA_URL}...")
            # SSL context handling for environments with missing root certificates
            ctx = ssl.create_default_context()
            try:
                # First try with verified certificate
                req = urllib.request.Request(
                    self.DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(
                    zip_path, "wb"
                ) as out_file:
                    out_file.write(response.read())
            except (ssl.SSLCertVerificationError, urllib.error.URLError):
                # Fallback to unverified context if local CA certs are missing
                unverified_ctx = ssl._create_unverified_context()
                req = urllib.request.Request(
                    self.DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(
                    req, context=unverified_ctx, timeout=30
                ) as response, open(zip_path, "wb") as out_file:
                    out_file.write(response.read())

            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            if zip_path.exists():
                try:
                    zip_path.unlink()
                except OSError:
                    pass

            print(f"Dataset extracted successfully to {self.dataset_dir}")
            return self.dataset_dir

        except Exception as e:
            if self.is_dataset_available():
                print(
                    f"Warning: Download failed ({e}), but local dataset files were found and will be used."
                )
                return self.dataset_dir
            raise RuntimeError(
                f"Failed to automatically download MovieLens dataset: {e}\n\n"
                f"MANUAL INSTRUCTIONS:\n"
                f"1. Download 'ml-latest-small.zip' from: {self.DATA_URL}\n"
                f"2. Extract the archive into: {self.data_dir.resolve()}\n"
                f"3. Ensure the folder '{self.DATA_FOLDER_NAME}' contains movies.csv and ratings.csv."
            ) from e

    def load_data(
        self, auto_download: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load, validate, clean, and enrich the MovieLens data.

        Returns:
            Tuple of (movies_df, ratings_df, tags_df, links_df)
        """
        if not self.is_dataset_available() and auto_download:
            self.download_and_extract()

        movies_path = self.dataset_dir / "movies.csv"
        ratings_path = self.dataset_dir / "ratings.csv"
        tags_path = self.dataset_dir / "tags.csv"
        links_path = self.dataset_dir / "links.csv"

        if not movies_path.exists() or not ratings_path.exists():
            raise FileNotFoundError(
                f"Missing required dataset files in {self.dataset_dir}. "
                "Please run download_and_extract() or provide dataset manually."
            )

        # 1. Load raw dataframes
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        tags_df = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame(
            columns=["userId", "movieId", "tag", "timestamp"]
        )
        links_df = pd.read_csv(links_path) if links_path.exists() else pd.DataFrame(
            columns=["movieId", "imdbId", "tmdbId"]
        )

        # 2. Data Cleaning
        # Deduplicate
        movies_df = movies_df.drop_duplicates(subset=["movieId"]).copy()
        ratings_df = ratings_df.drop_duplicates(subset=["userId", "movieId"]).copy()

        # Handle missing or malformed values
        movies_df["genres"] = movies_df["genres"].fillna("(no genres listed)")
        movies_df["title"] = movies_df["title"].fillna("Unknown Title").str.strip()

        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"]).copy()
        ratings_df["userId"] = ratings_df["userId"].astype(int)
        ratings_df["movieId"] = ratings_df["movieId"].astype(int)
        ratings_df["rating"] = ratings_df["rating"].astype(float)

        # 3. Extract release year from title e.g. "Toy Story (1995)" -> 1995
        def extract_year(title: str) -> Optional[int]:
            match = re.search(r"\((\d{4})\)", title)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
            return None

        movies_df["year"] = movies_df["title"].apply(extract_year)

        # Clean title without year for display
        movies_df["clean_title"] = movies_df["title"].apply(
            lambda t: re.sub(r"\s*\(\d{4}\)\s*", "", str(t)).strip() if pd.notna(t) else "Unknown Title"
        )

        # 4. Enrich metadata with user tags
        if not tags_df.empty:
            tags_df["tag"] = tags_df["tag"].fillna("").astype(str).str.lower().str.strip()
            # Aggregate tags per movie
            movie_tags = (
                tags_df[tags_df["tag"] != ""]
                .groupby("movieId")["tag"]
                .apply(lambda tags: " ".join(tags.unique()))
                .reset_index()
            )
            movies_df = movies_df.merge(movie_tags, on="movieId", how="left")
            movies_df["tag"] = movies_df["tag"].fillna("")
        else:
            movies_df["tag"] = ""

        # 5. Create Metadata Soup for Content-Based Filtering
        # Combine genres (replace | with space) and user tags
        def make_soup(row: pd.Series) -> str:
            genre_str = row["genres"].replace("|", " ").replace("-", "").lower()
            if genre_str == "(no genres listed)":
                genre_str = ""
            tag_str = row["tag"]
            # Repeat genre words to give them appropriate weight relative to unstructured tags
            return f"{genre_str} {genre_str} {tag_str}".strip()

        movies_df["metadata_soup"] = movies_df.apply(make_soup, axis=1)

        # Cache
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.tags_df = tags_df
        self.links_df = links_df

        return self.movies_df, self.ratings_df, self.tags_df, self.links_df

    def get_statistics(self) -> Dict[str, Any]:
        """Compute key summary statistics of the dataset."""
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()

        assert self.movies_df is not None
        assert self.ratings_df is not None

        num_movies = len(self.movies_df)
        num_users = self.ratings_df["userId"].nunique()
        num_ratings = len(self.ratings_df)

        # Matrix sparsity
        total_possible = num_movies * num_users
        sparsity = (1.0 - (num_ratings / total_possible)) * 100 if total_possible > 0 else 0.0

        # Unique genres
        all_genres = set()
        for g_list in self.movies_df["genres"].str.split("|"):
            for g in g_list:
                if g and g != "(no genres listed)":
                    all_genres.add(g)

        ratings_series = self.ratings_df["rating"]

        return {
            "num_movies": num_movies,
            "num_users": num_users,
            "num_ratings": num_ratings,
            "num_genres": len(all_genres),
            "genres_list": sorted(list(all_genres)),
            "sparsity_pct": round(sparsity, 2),
            "rating_min": float(ratings_series.min()),
            "rating_max": float(ratings_series.max()),
            "rating_mean": round(float(ratings_series.mean()), 2),
            "rating_median": float(ratings_series.median()),
            "avg_ratings_per_user": round(num_ratings / max(num_users, 1), 1),
            "avg_ratings_per_movie": round(num_ratings / max(num_movies, 1), 1),
        }

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Get all ratings made by a specific user with movie metadata."""
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()

        assert self.movies_df is not None
        assert self.ratings_df is not None

        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame()

        cols_to_merge = ["movieId", "title", "clean_title", "genres", "year"]
        avail_cols = [c for c in cols_to_merge if c in self.movies_df.columns]
        merged = user_ratings.merge(
            self.movies_df[avail_cols],
            on="movieId",
            how="left",
        )
        if "clean_title" not in merged.columns and "title" in merged.columns:
            merged["clean_title"] = merged["title"].apply(
                lambda t: re.sub(r"\s*\(\d{4}\)\s*", "", str(t)).strip() if pd.notna(t) else "Unknown Title"
            )
        return merged.sort_values(by="rating", ascending=False).reset_index(drop=True)

    def get_popular_movies(self, top_n: int = 10, min_ratings: int = 20) -> pd.DataFrame:
        """Get top popular movies using IMDb's Bayesian Weighted Rating formula.

        Formula:
            WR = (v / (v + m)) * R + (m / (v + m)) * C
        where:
            v = number of ratings for the movie
            m = minimum ratings threshold
            R = average rating of the movie
            C = overall average rating across all movies
        """
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()

        assert self.movies_df is not None
        assert self.ratings_df is not None

        grouped = (
            self.ratings_df.groupby("movieId")
            .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"))
            .reset_index()
        )

        overall_mean = self.ratings_df["rating"].mean()
        m = min_ratings

        qualified = grouped[grouped["rating_count"] >= m].copy()
        if qualified.empty:
            qualified = grouped.copy()
            m = 1

        v = qualified["rating_count"]
        r = qualified["rating_mean"]
        c = overall_mean

        qualified["weighted_score"] = (v / (v + m)) * r + (m / (v + m)) * c
        top_movies = qualified.sort_values(by="weighted_score", ascending=False).head(top_n)

        cols_to_merge = ["movieId", "title", "clean_title", "genres", "year"]
        avail_cols = [c for c in cols_to_merge if c in self.movies_df.columns]
        result = top_movies.merge(
            self.movies_df[avail_cols],
            on="movieId",
            how="left",
        )
        if "clean_title" not in result.columns and "title" in result.columns:
            result["clean_title"] = result["title"].apply(
                lambda t: re.sub(r"\s*\(\d{4}\)\s*", "", str(t)).strip() if pd.notna(t) else "Unknown Title"
            )
        result["rating_mean"] = result["rating_mean"].round(2)
        result["weighted_score"] = result["weighted_score"].round(2)
        return result.reset_index(drop=True)

    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata and rating summary for a single movie."""
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()

        assert self.movies_df is not None
        assert self.ratings_df is not None

        movie_row = self.movies_df[self.movies_df["movieId"] == movie_id]
        if movie_row.empty:
            return None

        m_ratings = self.ratings_df[self.ratings_df["movieId"] == movie_id]
        count = len(m_ratings)
        avg_rating = round(float(m_ratings["rating"].mean()), 2) if count > 0 else 0.0

        row = movie_row.iloc[0]
        return {
            "movieId": int(row["movieId"]),
            "title": str(row["title"]),
            "clean_title": str(row["clean_title"]),
            "genres": str(row["genres"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating_count": count,
            "avg_rating": avg_rating,
        }
