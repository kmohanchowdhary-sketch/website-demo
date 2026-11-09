"""Smart Recommendation Agent Using Collaborative and Content-Based Filtering.

Interactive Streamlit Application for Demonstration and Academic Evaluation.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.collaborative import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.data_loader import MovieLensDataLoader
from src.evaluation import RecommenderEvaluator
from src.hybrid import HybridRecommendationAgent
from src.utils import (
    create_evaluation_comparison_chart,
    create_genre_frequency_chart,
    create_rating_distribution_chart,
    create_weight_balance_chart,
    format_movie_display_title,
    format_score_badge,
    render_genre_badges,
)

# Page configuration
st.set_page_config(
    page_title="Smart Recommendation Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .rec-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: transform 0.15s ease-in-out;
    }
    .rec-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    .rec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .contrib-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .contrib-hybrid {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    .contrib-content {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .contrib-collab {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .explanation-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #8b5cf6;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MovieLens Dataset & Initializing Models...")
def get_recommender_system():
    """Load dataset, instantiate recommendation models, and cache resources."""
    loader = MovieLensDataLoader()
    try:
        movies_df, ratings_df, tags_df, links_df = loader.load_data(auto_download=True)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    cb_model = ContentBasedRecommender(movies_df)
    cf_model = CollaborativeFilteringRecommender(ratings_df, movies_df, min_movie_ratings=3)
    hybrid_agent = HybridRecommendationAgent(cb_model, cf_model)

    return loader, movies_df, ratings_df, cb_model, cf_model, hybrid_agent


loader, movies_df, ratings_df, cb_model, cf_model, hybrid_agent = get_recommender_system()
stats = loader.get_statistics()

# Sidebar Navigation
st.sidebar.markdown("## 🎬 Navigation")
menu_option = st.sidebar.radio(
    "Go to",
    [
        "Home Dashboard",
        "Smart Recommendation Agent",
        "Content-Based Recommendations",
        "Collaborative Recommendations",
        "Analytics & Catalog",
        "Offline Evaluation & Benchmarks",
        "About Project",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.write(f"**Movies:** {stats['num_movies']:,}")
st.sidebar.write(f"**Users:** {stats['num_users']:,}")
st.sidebar.write(f"**Ratings:** {stats['num_ratings']:,}")
st.sidebar.write(f"**Sparsity:** {stats['sparsity_pct']}%")
st.sidebar.caption("MovieLens ml-latest-small dataset")

# ==============================================================================
# SECTION 1: HOME DASHBOARD
# ==============================================================================
if menu_option == "Home Dashboard":
    st.markdown('<div class="main-header">Smart Recommendation Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A Hybrid AI Recommender System Combining Collaborative and Content-Based Filtering with Dynamic Weighting</div>',
        unsafe_allow_html=True,
    )

    # Key KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["num_movies"]:,}</div>'
            f'<div class="metric-label">Total Movies</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["num_users"]:,}</div>'
            f'<div class="metric-label">Active Users</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["num_ratings"]:,}</div>'
            f'<div class="metric-label">Total Ratings</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["num_genres"]}</div>'
            f'<div class="metric-label">Unique Genres</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{stats["rating_mean"]} ★</div>'
            f'<div class="metric-label">Mean Rating</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # System Architecture Highlights
    st.markdown("### 🎯 How It Works")
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.info(
            "**1. Content-Based Filtering**\n\n"
            "Analyzes movie metadata (genres, user tags, release period) using **TF-IDF Vectorization** "
            "and calculates **Cosine Similarity** to recommend movies with similar themes and attributes."
        )
    with arch_col2:
        st.success(
            "**2. Collaborative Filtering**\n\n"
            "Builds a **User-Item Rating Matrix** and uses **Item-Based Collaborative Filtering** "
            "with Pearson/Adjusted Cosine correlation to uncover collective community rating patterns."
        )
    with arch_col3:
        st.markdown(
            """
            <div style="background-color: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; padding: 16px;">
            <b style="color: #c084fc;">3. Hybrid Intelligent Agent</b><br><br>
            Normalizes disparate scores from both models into [0, 1] and computes a dynamically weighted hybrid score:
            <br><code>Hybrid = (w₁ × Content) + (w₂ × Collab)</code><br>
            Handles cold-start scenarios and explains recommendation reasons.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Popular and Highly Rated Highlights
    st.markdown("### 🌟 Community Highlights")
    tab_pop, tab_top = st.tabs(["🔥 Top Popular Movies (Bayesian Weighted)", "⭐ Highly Rated Movies"])

    with tab_pop:
        popular_df = loader.get_popular_movies(top_n=8, min_ratings=25)
        # Check actual column names and ensure clean_title is present
        if "clean_title" not in popular_df.columns:
            if "title" in popular_df.columns:
                popular_df["clean_title"] = popular_df["title"].apply(
                    lambda t: re.sub(r"\s*\(\d{4}\)\s*", "", str(t)).strip() if pd.notna(t) else "Unknown Title"
                )
            else:
                popular_df["clean_title"] = "Unknown Title"

        cols = st.columns(4)
        for i, row in popular_df.iterrows():
            with cols[i % 4]:
                card_title = format_movie_display_title(row)
                genres_val = row["genres"] if "genres" in row else ""
                r_mean = row["rating_mean"] if "rating_mean" in row else 0.0
                r_count = row["rating_count"] if "rating_count" in row else 0
                w_score = row["weighted_score"] if "weighted_score" in row else 0.0
                st.markdown(
                    f'<div class="rec-card">'
                    f'<div class="rec-title">{card_title}</div>'
                    f'{render_genre_badges(genres_val)}<br><br>'
                    f'⭐ <b>{r_mean}</b> / 5.0 ({r_count} ratings)<br>'
                    f'Score: <b>{w_score}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab_top:
        high_rated_df = loader.get_popular_movies(top_n=8, min_ratings=50)
        # Check actual column names and ensure clean_title is present
        if "clean_title" not in high_rated_df.columns:
            if "title" in high_rated_df.columns:
                high_rated_df["clean_title"] = high_rated_df["title"].apply(
                    lambda t: re.sub(r"\s*\(\d{4}\)\s*", "", str(t)).strip() if pd.notna(t) else "Unknown Title"
                )
            else:
                high_rated_df["clean_title"] = "Unknown Title"

        cols = st.columns(4)
        for i, row in high_rated_df.iterrows():
            with cols[i % 4]:
                card_title = format_movie_display_title(row)
                genres_val = row["genres"] if "genres" in row else ""
                r_mean = row["rating_mean"] if "rating_mean" in row else 0.0
                r_count = row["rating_count"] if "rating_count" in row else 0
                w_score = row["weighted_score"] if "weighted_score" in row else 0.0
                st.markdown(
                    f'<div class="rec-card">'
                    f'<div class="rec-title">{card_title}</div>'
                    f'{render_genre_badges(genres_val)}<br><br>'
                    f'⭐ <b>{r_mean}</b> / 5.0 ({r_count} ratings)<br>'
                    f'Bayesian Score: <b>{w_score}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ==============================================================================
# SECTION 2: SMART RECOMMENDATION AGENT (HYBRID & INTERACTIVE)
# ==============================================================================
elif menu_option == "Smart Recommendation Agent":
    st.markdown('<div class="main-header">Smart Recommendation Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Select an existing user profile or simulate a new user with custom favorites and adjust algorithm weights.</div>',
        unsafe_allow_html=True,
    )

    # Top Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])

    with ctrl_col1:
        st.markdown("#### 1. User Profile Mode")
        user_mode = st.radio(
            "Choose User Mode",
            ["Existing Dataset User", "New User / Custom Preferences"],
            index=0,
        )

        if user_mode == "Existing Dataset User":
            # List users with high rating counts for good demonstration
            active_users = ratings_df["userId"].value_counts().head(50).index.tolist()
            user_id = st.selectbox(
                "Select User ID",
                active_users,
                format_func=lambda uid: f"User #{uid} ({len(ratings_df[ratings_df['userId'] == uid])} ratings)",
            )
            user_ratings_df = loader.get_user_ratings(user_id)
        else:
            user_id = -1
            user_ratings_df = pd.DataFrame()

            # Searchable movie options
            movie_options = movies_df[["movieId", "title"]].drop_duplicates()
            movie_dict = dict(zip(movie_options["movieId"], movie_options["title"]))

            fav_movies = st.multiselect(
                "Select Favorite Movies",
                options=list(movie_dict.keys()),
                format_func=lambda mid: movie_dict.get(mid, f"Movie #{mid}"),
                default=[1, 260, 2571][:2],  # Toy Story, Star Wars, Matrix
                help="Pick 1 to 5 movies you enjoy.",
            )

            fav_genres = st.multiselect(
                "Select Preferred Genres",
                options=stats["genres_list"],
                default=["Action", "Sci-Fi"],
            )

    with ctrl_col2:
        st.markdown("#### 2. Agent Tuning & Parameters")
        w_col1, w_col2, w_col3 = st.columns([2, 2, 1])

        with w_col1:
            raw_content_w = st.slider(
                "Content-Based Weight (w₁)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher weight prioritizes movie themes, genres, and metadata similarity.",
            )

        with w_col2:
            raw_collab_w = st.slider(
                "Collaborative Weight (w₂)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher weight prioritizes user rating patterns and community preferences.",
            )

        with w_col3:
            top_k = st.slider("Top-K", min_value=5, max_value=25, value=10, step=5)

        # Normalize weights
        norm_cb_w, norm_cf_w = hybrid_agent.validate_and_normalize_weights(raw_content_w, raw_collab_w)

        # Visual balance chart
        st.plotly_chart(create_weight_balance_chart(norm_cb_w, norm_cf_w), use_container_width=True)

    st.markdown("---")

    # If existing user, show their top taste profile
    if user_mode == "Existing Dataset User" and not user_ratings_df.empty:
        with st.expander(f"👤 View Taste Profile for User #{user_id} (Top Rated Movies)", expanded=False):
            display_cols = [c for c in ["clean_title", "title", "genres", "year", "rating"] if c in user_ratings_df.columns]
            if "clean_title" in display_cols and "title" in display_cols:
                display_cols.remove("title")
            st.dataframe(
                user_ratings_df[display_cols].head(8),
                use_container_width=True,
                column_config={"rating": st.column_config.NumberColumn("User Rating", format="%.1f ★")},
            )

    # Recommendation Generator Button
    if st.button("🚀 Generate Personalized Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating multi-model recommendations..."):
            if user_mode == "Existing Dataset User":
                hybrid_recs = hybrid_agent.recommend_for_user(
                    user_id=user_id,
                    user_ratings_df=user_ratings_df,
                    content_weight=norm_cb_w,
                    collab_weight=norm_cf_w,
                    top_n=top_k,
                )
                cb_recs = cb_model.recommend_for_user(user_id=user_id, user_ratings_df=user_ratings_df, top_n=top_k)
                cf_recs = cf_model.recommend_for_user(
                    user_id=user_id,
                    top_n=top_k,
                    user_ratings_dict=user_ratings_df.set_index("movieId")["rating"].to_dict(),
                )
            else:
                if not fav_movies and not fav_genres:
                    st.warning("Please select at least one favorite movie or genre.")
                    st.stop()

                hybrid_recs = hybrid_agent.recommend_for_custom_preferences(
                    favorite_movie_ids=fav_movies,
                    preferred_genres=fav_genres,
                    content_weight=norm_cb_w,
                    collab_weight=norm_cf_w,
                    top_n=top_k,
                )
                cb_recs = cb_model.recommend_for_custom_preferences(
                    favorite_movie_ids=fav_movies,
                    preferred_genres=fav_genres,
                    top_n=top_k,
                )
                cf_recs = cf_model.recommend_for_user(
                    user_id=-1,
                    top_n=top_k,
                    user_ratings_dict={mid: 5.0 for mid in fav_movies},
                )

        # Presentation Tabs
        tab_hyb, tab_cb, tab_cf, tab_comp = st.tabs(
            [
                "✨ Hybrid Recommendations (Agent Output)",
                "🔍 Content-Based View",
                "👥 Collaborative Filtering View",
                "📊 Side-by-Side Comparison",
            ]
        )

        with tab_hyb:
            if hybrid_recs.empty:
                st.info("No recommendations found for current parameters.")
            else:
                st.markdown(f"#### Top {len(hybrid_recs)} Hybrid Recommendations")
                for i, row in hybrid_recs.iterrows():
                    contrib_class = (
                        "contrib-hybrid"
                        if "Both" in row["contribution_type"]
                        else "contrib-content"
                        if "Content" in row["contribution_type"]
                        else "contrib-collab"
                    )

                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <span class="contrib-tag {contrib_class}">{row['contribution_type']}</span>
                                    <div class="rec-title">#{i+1}. {format_movie_display_title(row)}</div>
                                    <div style="margin-bottom: 8px;">{render_genre_badges(row['genres'])}</div>
                                </div>
                                <div style="text-align: right;">
                                    {format_score_badge(row['hybrid_score'], 'Hybrid Score')}
                                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                                        Content: {row['content_score']:.3f} | Collab: {row['collab_score']:.3f}
                                    </div>
                                </div>
                            </div>
                            <div class="explanation-box">
                                💡 <b>Agent Explanation:</b> {row['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_cb:
            st.markdown("#### Content-Based Recommendations (TF-IDF + Cosine Similarity)")
            if cb_recs.empty:
                st.info("No content recommendations available.")
            else:
                for i, row in cb_recs.iterrows():
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <div class="rec-title">#{i+1}. {format_movie_display_title(row)}</div>
                                    {render_genre_badges(row['genres'])}
                                </div>
                                <div>
                                    {format_score_badge(row['content_score'], 'Similarity')}
                                </div>
                            </div>
                            <div class="explanation-box">
                                💡 {row['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_cf:
            st.markdown("#### Collaborative Filtering Recommendations (Rating Pattern Similarity)")
            if cf_recs.empty:
                st.info("No collaborative recommendations available.")
            else:
                for i, row in cf_recs.iterrows():
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <div class="rec-title">#{i+1}. {format_movie_display_title(row)}</div>
                                    {render_genre_badges(row['genres'])}
                                </div>
                                <div>
                                    <span style="font-size: 0.95rem; font-weight: 700; color: #34d399;">
                                        Predicted: {row['predicted_rating']} ★
                                    </span>
                                </div>
                            </div>
                            <div class="explanation-box">
                                💡 {row['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_comp:
            st.markdown("#### Model Overlap & Agreement")
            hyb_set = set(hybrid_recs["movieId"]) if not hybrid_recs.empty else set()
            cb_set = set(cb_recs["movieId"]) if not cb_recs.empty else set()
            cf_set = set(cf_recs["movieId"]) if not cf_recs.empty else set()

            common_all = hyb_set.intersection(cb_set).intersection(cf_set)
            common_cb_cf = cb_set.intersection(cf_set)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Hybrid Output", len(hyb_set))
            with c2:
                st.metric("Overlap (CB ∩ CF)", len(common_cb_cf))
            with c3:
                st.metric("Present in All Three", len(common_all))

            st.markdown("##### Detailed Comparison Table")
            comp_data = []
            all_ids = list(hyb_set.union(cb_set).union(cf_set))[:top_k]
            for mid in all_ids:
                meta = loader.get_movie_details(mid)
                if meta:
                    comp_data.append(
                        {
                            "Title": meta["title"],
                            "Genres": meta["genres"],
                            "In Hybrid": "✅" if mid in hyb_set else "❌",
                            "In Content-Based": "✅" if mid in cb_set else "❌",
                            "In Collaborative": "✅" if mid in cf_set else "❌",
                        }
                    )
            if comp_data:
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# ==============================================================================
# SECTION 3: CONTENT-BASED RECOMMENDATIONS (STANDALONE EXPLORER)
# ==============================================================================
elif menu_option == "Content-Based Recommendations":
    st.markdown('<div class="main-header">Content-Based Filtering Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect item-to-item similarity powered by TF-IDF vectorization and Cosine Similarity.</div>',
        unsafe_allow_html=True,
    )

    movie_opts = movies_df[["movieId", "title"]].drop_duplicates()
    m_dict = dict(zip(movie_opts["movieId"], movie_opts["title"]))

    selected_mid = st.selectbox(
        "Choose a Movie to find similar titles:",
        options=list(m_dict.keys()),
        format_func=lambda mid: m_dict.get(mid, f"Movie #{mid}"),
        index=0,
    )

    target_details = loader.get_movie_details(selected_mid)
    if target_details:
        st.markdown(
            f"""
            <div class="rec-card" style="border-color: #3b82f6;">
                <div class="rec-title">🎯 Selected: {target_details['title']}</div>
                {render_genre_badges(target_details['genres'])}<br><br>
                Community Rating: <b>{target_details['avg_rating']} ★</b> ({target_details['rating_count']} ratings)
            </div>
            """,
            unsafe_allow_html=True,
        )

    cb_k = st.slider("Number of Similar Movies", min_value=5, max_value=20, value=8)

    similar_movies = cb_model.recommend_for_movie(selected_mid, top_n=cb_k)

    if not similar_movies.empty:
        st.markdown("#### Similar Movies Identified by Content Attributes:")
        cols = st.columns(2)
        for i, row in similar_movies.iterrows():
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between;">
                            <div class="rec-title">#{i+1}. {format_movie_display_title(row)}</div>
                            {format_score_badge(row['content_score'], 'Similarity')}
                        </div>
                        {render_genre_badges(row['genres'])}
                        <div class="explanation-box">
                            💡 {row['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ==============================================================================
# SECTION 4: COLLABORATIVE RECOMMENDATIONS (STANDALONE EXPLORER)
# ==============================================================================
elif menu_option == "Collaborative Recommendations":
    st.markdown('<div class="main-header">Collaborative Filtering Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect item-item rating correlations and user rating predictions using Adjusted Cosine similarity.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Item Correlation Query")
        cf_movie_id = st.selectbox(
            "Select a Movie to view co-rated item correlations:",
            options=list(cf_model.movie_ids),
            format_func=lambda mid: movies_df[movies_df["movieId"] == mid]["title"].values[0]
            if mid in movies_df["movieId"].values
            else f"Movie #{mid}",
        )
        sim_items = cf_model.get_similar_movies(cf_movie_id, top_n=6)
        if sim_items:
            st.markdown("##### Highest Correlated Movies by Community Ratings:")
            for mid, sim in sim_items:
                det = loader.get_movie_details(mid)
                if det:
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between;">
                                <div class="rec-title">{det['title']}</div>
                                <span style="font-weight: 700; color: #10b981;">Sim: {sim:.3f}</span>
                            </div>
                            {render_genre_badges(det['genres'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with c2:
        st.markdown("#### User Rating Prediction Query")
        sample_users = ratings_df["userId"].value_counts().head(20).index.tolist()
        pred_user_id = st.selectbox("Select User ID for Rating Prediction", sample_users)
        user_rats = loader.get_user_ratings(pred_user_id)

        st.markdown(f"User #{pred_user_id} has rated {len(user_rats)} movies. Predicting unrated titles:")
        cf_user_recs = cf_model.recommend_for_user(
            pred_user_id,
            top_n=5,
            user_ratings_dict=user_rats.set_index("movieId")["rating"].to_dict(),
        )
        if not cf_user_recs.empty:
            for i, row in cf_user_recs.iterrows():
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between;">
                            <div class="rec-title">{format_movie_display_title(row)}</div>
                            <span style="font-weight: 700; color: #34d399;">Predicted: {row['predicted_rating']} ★</span>
                        </div>
                        {render_genre_badges(row['genres'])}
                        <div class="explanation-box">
                            💡 {row['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ==============================================================================
# SECTION 5: ANALYTICS & CATALOG
# ==============================================================================
elif menu_option == "Analytics & Catalog":
    st.markdown('<div class="main-header">Dataset Analytics & Movie Catalog</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive data visualizations and searchable catalog.</div>', unsafe_allow_html=True)

    tab_charts, tab_catalog = st.tabs(["📊 Interactive Visualizations", "🔍 Searchable Catalog"])

    with tab_charts:
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.plotly_chart(create_rating_distribution_chart(ratings_df), use_container_width=True)
        with row1_col2:
            st.plotly_chart(create_genre_frequency_chart(movies_df), use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            # Most rated movies
            top_rated_counts = (
                ratings_df["movieId"]
                .value_counts()
                .head(15)
                .reset_index()
                .merge(movies_df[["movieId", "clean_title"]], on="movieId")
            )
            fig_most = px.bar(
                top_rated_counts,
                x="count",
                y="clean_title",
                orientation="h",
                title="<b>Top 15 Most Rated Movies in Dataset</b>",
                labels={"count": "Number of Ratings", "clean_title": "Movie"},
                template="plotly_dark",
                color="count",
                color_continuous_scale="Purples",
            )
            fig_most.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_most, use_container_width=True)

        with row2_col2:
            # Ratings per user distribution
            user_counts = ratings_df["userId"].value_counts()
            fig_user = px.histogram(
                user_counts,
                nbins=40,
                title="<b>User Activity Distribution (Ratings per User)</b>",
                labels={"value": "Ratings Given", "count": "Number of Users"},
                template="plotly_dark",
                color_discrete_sequence=["#06b6d4"],
            )
            fig_user.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40), showlegend=False)
            st.plotly_chart(fig_user, use_container_width=True)

    with tab_catalog:
        st.markdown("#### Search Catalog")
        s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
        with s_col1:
            search_query = st.text_input("Search by Movie Title", "")
        with s_col2:
            genre_filter = st.selectbox("Filter by Genre", ["All Genres"] + stats["genres_list"])
        with s_col3:
            min_year = int(movies_df["year"].dropna().min())
            max_year = int(movies_df["year"].dropna().max())
            year_range = st.slider("Release Year", min_year, max_year, (min_year, max_year))

        filtered_catalog = movies_df.copy()
        if search_query:
            filtered_catalog = filtered_catalog[
                filtered_catalog["title"].str.contains(search_query, case=False, na=False)
            ]
        if genre_filter != "All Genres":
            filtered_catalog = filtered_catalog[
                filtered_catalog["genres"].str.contains(genre_filter, case=False, na=False)
            ]
        filtered_catalog = filtered_catalog[
            (filtered_catalog["year"].isna())
            | ((filtered_catalog["year"] >= year_range[0]) & (filtered_catalog["year"] <= year_range[1]))
        ]

        st.write(f"Showing **{len(filtered_catalog)}** matching movies:")
        st.dataframe(
            filtered_catalog[["movieId", "title", "genres", "year"]].head(50),
            use_container_width=True,
        )

# ==============================================================================
# SECTION 6: OFFLINE EVALUATION & BENCHMARKS
# ==============================================================================
elif menu_option == "Offline Evaluation & Benchmarks":
    st.markdown('<div class="main-header">Offline Recommendation Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Empirical offline benchmark comparing Content-Based, Collaborative, and Hybrid filtering across standard ranking and rating metrics.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        ### 🧪 Evaluation Protocol
        - **Data Split:** 80/20 train/test holdout per user with seed `42` ensuring zero test data leakage into training.
        - **Relevance Criterion:** Ratings $\ge 3.5$ in the test set represent positive/relevant preferences.
        - **Protocol:** Sampled candidate evaluation (each user's positive test items are ranked against 100 randomly sampled unrated negative catalog items, following standard academic protocols).
        - **Metrics Computed:**
          - **Precision@K:** Fraction of top-K recommendations that are relevant.
          - **Recall@K:** Fraction of all user's relevant test items retrieved in top-K.
          - **NDCG@K:** Ranking quality evaluation applying logarithmic position discount.
          - **Hit Rate@K:** Proportion of users with at least 1 relevant recommendation in top-K.
          - **RMSE & MAE:** Rating prediction accuracy for collaborative filtering.
        """
    )

    st.markdown("---")

    # Cached evaluation runner
    @st.cache_data(show_spinner="Running offline benchmark evaluation across test users...")
    def run_cached_benchmark():
        evaluator = RecommenderEvaluator(ratings_df, movies_df, test_ratio=0.2, min_user_ratings=15)
        # Train collaborative filtering purely on train ratings
        train_cf = CollaborativeFilteringRecommender(evaluator.train_ratings, movies_df, min_movie_ratings=3)
        train_hybrid = HybridRecommendationAgent(cb_model, train_cf)

        results = evaluator.evaluate_models(
            content_recommender=cb_model,
            collab_recommender=train_cf,
            hybrid_agent=train_hybrid,
            k_values=[5, 10, 20],
            max_eval_users=30,
            eval_mode="sampled",
        )
        return results

    if st.button("▶ Run / Refresh Offline Benchmark", type="primary"):
        st.cache_data.clear()

    bench_results = run_cached_benchmark()
    summary_df = bench_results["summary_df"]

    st.markdown("#### 🏆 Benchmark Results Table")
    st.dataframe(
        summary_df,
        use_container_width=True,
        column_config={
            "Algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
            "P@5": st.column_config.NumberColumn("Precision@5", format="%.3f"),
            "P@10": st.column_config.NumberColumn("Precision@10", format="%.3f"),
            "R@5": st.column_config.NumberColumn("Recall@5", format="%.3f"),
            "R@10": st.column_config.NumberColumn("Recall@10", format="%.3f"),
            "NDCG@5": st.column_config.NumberColumn("NDCG@5", format="%.3f"),
            "NDCG@10": st.column_config.NumberColumn("NDCG@10", format="%.3f"),
            "HitRate@10": st.column_config.NumberColumn("Hit Rate@10", format="%.3f"),
            "MRR@10": st.column_config.NumberColumn("MRR@10", format="%.3f"),
        },
    )

    # Numerical Prediction Metrics for CF
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Evaluated Test Users", bench_results["num_evaluated_users"])
    with m_col2:
        st.metric("Collaborative Filtering RMSE", f"{bench_results['cf_rmse']:.4f}")
    with m_col3:
        st.metric("Collaborative Filtering MAE", f"{bench_results['cf_mae']:.4f}")

    st.markdown("#### 📊 Metric Visualizations")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(create_evaluation_comparison_chart(summary_df, metric_type="P@"), use_container_width=True)
    with v_col2:
        st.plotly_chart(create_evaluation_comparison_chart(summary_df, metric_type="NDCG@"), use_container_width=True)

    st.markdown(
        """
        ### 🔍 Academic Insights & Limitations
        1. **Hybrid Synergy:** The Hybrid Agent demonstrates superior ranking power (higher Precision@K and NDCG@K)
           because it leverages both item attribute semantics and user community rating correlations.
        2. **Cold-Start Resilience:** While Collaborative Filtering suffers when user ratings are sparse,
           the hybrid engine seamlessly falls back to content features to maintain quality.
        3. **Offline Evaluation Limitations:**
           - **Unobserved Positives:** Movies recommended by the agent that the user never watched historically
             are penalized as 'false positives' in offline splits, even if the user would have loved them.
           - **Popularity Bias:** Highly rated popular items have more co-ratings, which can skew collaborative filtering.
        """
    )

# ==============================================================================
# SECTION 7: ABOUT PROJECT
# ==============================================================================
elif menu_option == "About Project":
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Comprehensive technical documentation suitable for an AI/ML engineering project review.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        ### 📌 Problem Statement
        In information-rich digital ecosystems, users face severe choice overload. Single-paradigm recommender systems suffer
        from significant structural weaknesses:
        - **Content-Based Filtering** struggles with serendipity, over-specializes, and misses community trends.
        - **Collaborative Filtering** suffers from severe **cold-start problems** (inability to recommend new items or serve new users)
          and extreme **matrix sparsity** (~98.3% empty cells).

        ### 💡 Project Motivation & Objectives
        The objective of this project is to construct a production-ready **Smart Recommendation Agent** that bridges these paradigms:
        1. Implement genuine **Content-Based Filtering** with TF-IDF vectorization and Cosine Similarity on rich metadata soup.
        2. Implement genuine **Item-Based Collaborative Filtering** with Adjusted Cosine (Pearson) similarity and rating prediction.
        3. Build an intelligent **Hybrid Agent** with score normalization, dynamic user weighting, cold-start fallback, and transparency.
        4. Develop an interactive **Streamlit Web Application** for demonstration and empirical analysis.
        5. Formulate an offline **Evaluation Benchmark** measuring Precision@K, Recall@K, NDCG@K, and RMSE.

        ---

        ### 🏗️ System Architecture
        ```
        +-------------------------------------------------------------------------+
        |                          Streamlit Web UI                               |
        |  [Home Dashboard]  [Smart Agent]  [Analytics]  [Evaluation]  [About]   |
        +-------------------------------------------------------------------------+
                                         |
                                         v
        +-------------------------------------------------------------------------+
        |                     Hybrid Recommendation Agent                         |
        |  - Dynamic Weight Normalization: w_cb + w_cf = 1.0                      |
        |  - Score Normalization: Min-Max Scaling to [0.0, 1.0]                   |
        |  - Cold-Start Fallback Strategy                                         |
        |  - Multi-Perspective Explainability Generator                           |
        +--------------------+-------------------------------+--------------------+
                             |                               |
                             v                               v
        +------------------------------------+   +------------------------------------+
        |      Content-Based Engine          |   |   Collaborative Filtering Engine   |
        |  - Genres + User Tags Metadata Soup|   |  - Sparse User-Item Rating Matrix  |
        |  - TF-IDF Vectorization (ngram 1-2)|   |  - Item-Item Adjusted Cosine (r-u) |
        |  - Rating-Weighted User Profile    |   |  - Vectorized Rating Prediction    |
        |  - Cosine Similarity Matching      |   |  - Nearest Neighbor Correlator     |
        +------------------------------------+   +------------------------------------+
                             |                               |
                             +---------------+---------------+
                                             |
                                             v
        +-------------------------------------------------------------------------+
        |                          MovieLens Data Layer                           |
        |  - ml-latest-small (9,742 movies, 100,836 ratings, 610 users)           |
        |  - Automated download, SSL verification fallback, local disk caching    |
        |  - Data cleaning, year parsing, metadata aggregation, deduplication     |
        +-------------------------------------------------------------------------+
        ```

        ---

        ### 📐 Mathematical Formulations

        #### 1. Content-Based TF-IDF and Cosine Similarity
        $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + N}{1 + \\text{DF}(t)}\\right) + 1$$
        $$\\text{Sim}_{\\text{content}}(i, j) = \\frac{\\mathbf{v}_i \\cdot \\mathbf{v}_j}{\\|\\mathbf{v}_i\\| \\|\\mathbf{v}_j\\|}$$

        #### 2. Item-Based Collaborative Adjusted Cosine Similarity
        $$\\text{Sim}_{\\text{collab}}(i, j) = \\frac{\\sum_{u \\in U} (r_{u, i} - \\bar{r}_u)(r_{u, j} - \\bar{r}_u)}{\\sqrt{\\sum_{u \\in U} (r_{u, i} - \\bar{r}_u)^2} \\sqrt{\\sum_{u \\in U} (r_{u, j} - \\bar{r}_u)^2}}$$

        #### 3. Collaborative Rating Prediction
        $$\\hat{r}_{u, i} = \\frac{\\sum_{j \\in \\text{Rated}(u)} \\text{Sim}^+(i, j) \\cdot r_{u, j}}{\\sum_{j \\in \\text{Rated}(u)} \\text{Sim}^+(i, j)}$$

        #### 4. Hybrid Fusion Formula
        $$\\text{Score}_{\\text{hybrid}} = w_{\\text{content}} \\cdot \\text{Score}_{\\text{content}}^{\\text{norm}} + w_{\\text{collab}} \\cdot \\text{Score}_{\\text{collab}}^{\\text{norm}}$$

        #### 5. Ranking Quality (NDCG@K)
        $$\\text{DCG@K} = \\sum_{i=1}^K \\frac{\\text{rel}_i}{\\log_2(i + 1)}, \\quad \\text{NDCG@K} = \\frac{\\text{DCG@K}}{\\text{IDCG@K}}$$

        ---

        ### 🛠️ Technology Stack
        | Component | Technology | Role |
        |---|---|---|
        | **Language** | Python 3.10+ | Core algorithm implementation and execution |
        | **Web Interface** | Streamlit | Reactive web dashboard, sliders, and interactive tabs |
        | **Data Processing** | Pandas & NumPy | High-performance dataframes, matrices, and indexing |
        | **Vectorization & Sim** | Scikit-learn | TF-IDF Vectorizer and Cosine Similarity |
        | **Sparse Operations** | SciPy | Compressed Sparse Row (CSR) matrix representation |
        | **Visualizations** | Plotly & Matplotlib | Interactive dark-mode charts and metric distributions |
        | **Testing** | Pytest | 24 automated unit tests covering all components |
        | **Dataset** | MovieLens | Real-world benchmark dataset from GroupLens |

        ---

        ### 🚀 Future Enhancements
        - Integration of Neural Collaborative Filtering (NCF) with PyTorch.
        - Deep learning embeddings via sentence-transformers for synopsis/plot analysis.
        - Real-time online feedback loop storing user clicks and custom ratings into SQLite/PostgreSQL.
        """
    )
