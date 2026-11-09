"""Unit tests for MovieLensDataLoader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import MovieLensDataLoader


@pytest.fixture
def sample_data(tmp_path):
    """Create a temporary mock MovieLens dataset for isolated testing."""
    ml_dir = tmp_path / "ml-latest-small"
    ml_dir.mkdir(parents=True)

    movies_csv = ml_dir / "movies.csv"
    movies_csv.write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        "2,Jumanji (1995),Adventure|Children|Fantasy\n"
        "3,Grumpier Old Men (1995),Comedy|Romance\n"
        "4,Heat (1995),Action|Crime|Thriller\n"
        "5,Sabrina (1995),Comedy|Romance\n"
    )

    ratings_csv = ml_dir / "ratings.csv"
    ratings_csv.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,4.0,964982703\n"
        "1,2,4.0,964981247\n"
        "1,3,4.0,964982224\n"
        "2,1,5.0,964982703\n"
        "2,4,4.5,964981247\n"
        "3,1,3.5,964982224\n"
        "3,5,4.0,964982224\n"
    )

    tags_csv = ml_dir / "tags.csv"
    tags_csv.write_text(
        "userId,movieId,tag,timestamp\n"
        "1,1,pixar,1139045764\n"
        "1,1,fun,1139045764\n"
        "2,4,action,1139045764\n"
    )

    return tmp_path


def test_initialization():
    loader = MovieLensDataLoader("custom_dir")
    assert str(loader.data_dir).endswith("custom_dir")


def test_load_data(sample_data):
    loader = MovieLensDataLoader(str(sample_data))
    movies, ratings, tags, links = loader.load_data(auto_download=False)

    assert len(movies) == 5
    assert len(ratings) == 7
    assert "year" in movies.columns
    assert "metadata_soup" in movies.columns

    # Year extraction test
    toy_story = movies[movies["movieId"] == 1].iloc[0]
    assert toy_story["year"] == 1995
    assert toy_story["clean_title"] == "Toy Story"
    # Metadata soup contains tag and genres
    assert "pixar" in toy_story["metadata_soup"]


def test_statistics(sample_data):
    loader = MovieLensDataLoader(str(sample_data))
    loader.load_data(auto_download=False)
    stats = loader.get_statistics()

    assert stats["num_movies"] == 5
    assert stats["num_users"] == 3
    assert stats["num_ratings"] == 7
    assert stats["rating_min"] == 3.5
    assert stats["rating_max"] == 5.0
    assert "Comedy" in stats["genres_list"]
    assert stats["sparsity_pct"] > 0


def test_get_user_ratings(sample_data):
    loader = MovieLensDataLoader(str(sample_data))
    loader.load_data(auto_download=False)

    u1_ratings = loader.get_user_ratings(1)
    assert len(u1_ratings) == 3
    assert "title" in u1_ratings.columns
    assert "clean_title" in u1_ratings.columns
    assert u1_ratings.iloc[0]["clean_title"] in ["Toy Story", "Jumanji", "Grumpier Old Men"]

    # Non-existent user
    assert loader.get_user_ratings(9999).empty


def test_get_popular_movies(sample_data):
    loader = MovieLensDataLoader(str(sample_data))
    loader.load_data(auto_download=False)

    pop = loader.get_popular_movies(top_n=3, min_ratings=1)
    assert len(pop) <= 3
    assert "weighted_score" in pop.columns
    assert "clean_title" in pop.columns
    assert "title" in pop.columns
    assert "year" in pop.columns


def test_get_movie_details(sample_data):
    loader = MovieLensDataLoader(str(sample_data))
    loader.load_data(auto_download=False)

    details = loader.get_movie_details(1)
    assert details is not None
    assert details["title"] == "Toy Story (1995)"
    assert details["clean_title"] == "Toy Story"
    assert details["rating_count"] == 3
    assert details["avg_rating"] > 0

    assert loader.get_movie_details(99999) is None


def test_format_movie_display_title():
    from src.utils import format_movie_display_title

    # Series with clean_title and float year
    s1 = pd.Series({"clean_title": "Toy Story", "year": 1995.0})
    assert format_movie_display_title(s1) == "Toy Story (1995)"

    # Series with integer year
    s2 = pd.Series({"clean_title": "Inception", "year": 2010})
    assert format_movie_display_title(s2) == "Inception (2010)"

    # Missing year
    s3 = pd.Series({"clean_title": "Unknown Title", "year": None})
    assert format_movie_display_title(s3) == "Unknown Title"

    # Fallback to title when clean_title is missing
    s4 = pd.Series({"title": "The Matrix (1999)"})
    assert format_movie_display_title(s4) == "The Matrix (1999)"

    # Dict format
    d1 = {"clean_title": "Interstellar", "year": 2014.0}
    assert format_movie_display_title(d1) == "Interstellar (2014)"
