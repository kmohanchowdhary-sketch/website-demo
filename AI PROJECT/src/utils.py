"""UI formatting, visualization helpers, and style utilities for Streamlit."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Genre color palette for attractive badges
GENRE_COLORS: Dict[str, str] = {
    "Action": "#ef4444",
    "Adventure": "#f97316",
    "Animation": "#8b5cf6",
    "Children": "#ec4899",
    "Comedy": "#eab308",
    "Crime": "#64748b",
    "Documentary": "#14b8a6",
    "Drama": "#3b82f6",
    "Fantasy": "#a855f7",
    "Film-Noir": "#1e293b",
    "Horror": "#dc2626",
    "Musical": "#d946ef",
    "Mystery": "#6366f1",
    "Romance": "#f43f5e",
    "Sci-Fi": "#06b6d4",
    "Thriller": "#0ea5e9",
    "War": "#78716c",
    "Western": "#b45309",
    "(no genres listed)": "#94a3b8",
}


def render_genre_badges(genres_str: str) -> str:
    """Convert a pipe-delimited genre string into styled HTML badges."""
    if not genres_str:
        return ""
    badges = []
    for g in genres_str.split("|"):
        g = g.strip()
        if not g:
            continue
        color = GENRE_COLORS.get(g, "#6b7280")
        badges.append(
            f'<span style="background-color: {color}20; color: {color}; '
            f'border: 1px solid {color}50; border-radius: 6px; padding: 2px 8px; '
            f'font-size: 0.75rem; font-weight: 600; margin-right: 5px; display: inline-block;">'
            f'{g}</span>'
        )
    return " ".join(badges)


def format_score_badge(score: float, label: str = "Score") -> str:
    """Format a numerical score into a highlighted pill."""
    pct = int(round(score * 100))
    if pct >= 80:
        bg = "#10b98120"
        fg = "#10b981"
        border = "#10b98150"
    elif pct >= 50:
        bg = "#3b82f620"
        fg = "#3b82f6"
        border = "#3b82f650"
    else:
        bg = "#f59e0b20"
        fg = "#f59e0b"
        border = "#f59e0b50"

    return (
        f'<span style="background-color: {bg}; color: {fg}; border: 1px solid {border}; '
        f'border-radius: 12px; padding: 3px 10px; font-size: 0.8rem; font-weight: 700;">'
        f'{label}: {score:.3f} ({pct}%)</span>'
    )


def format_movie_display_title(row: Any) -> str:
    """Format movie title and release year cleanly for UI cards and lists.

    Safely extracts the clean title (falling back to 'title') and appends the 4-digit
    year without float decimals (e.g. 'Toy Story (1995)'). Avoids duplicate years
    and handles missing columns or NaN values gracefully.
    """
    if hasattr(row, "get"):
        clean_title = row.get("clean_title", None)
        title = row.get("title", None)
        year = row.get("year", None)
    elif hasattr(row, "__getitem__"):
        clean_title = row["clean_title"] if "clean_title" in row else None
        title = row["title"] if "title" in row else None
        year = row["year"] if "year" in row else None
    else:
        clean_title = getattr(row, "clean_title", None)
        title = getattr(row, "title", None)
        year = getattr(row, "year", None)

    # Determine base movie title
    if clean_title is not None and pd.notna(clean_title) and str(clean_title).strip():
        base = str(clean_title).strip()
    elif title is not None and pd.notna(title) and str(title).strip():
        base = str(title).strip()
    else:
        base = "Unknown Movie"

    # Format release year if valid
    year_str = ""
    if year is not None and pd.notna(year) and year not in ("", "N/A"):
        try:
            year_int = int(float(year))
            if not base.endswith(f"({year_int})"):
                year_str = f" ({year_int})"
        except (ValueError, TypeError, OverflowError):
            pass

    return f"{base}{year_str}"


def create_rating_distribution_chart(ratings_df: pd.DataFrame) -> go.Figure:
    """Create interactive histogram of MovieLens user ratings."""
    counts = ratings_df["rating"].value_counts().sort_index()

    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(r) for r in counts.index],
                y=counts.values,
                marker=dict(
                    color=counts.values,
                    colorscale="Blues",
                    showscale=False,
                    line=dict(color="#1e3a8a", width=1),
                ),
                text=[f"{v:,}" for v in counts.values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="<b>Distribution of User Ratings</b>",
        xaxis_title="Rating (Stars)",
        yaxis_title="Total Count",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def create_genre_frequency_chart(movies_df: pd.DataFrame) -> go.Figure:
    """Create interactive bar chart of top movie genres."""
    genre_counts: Dict[str, int] = {}
    for g_list in movies_df["genres"].str.split("|"):
        for g in g_list:
            if g and g != "(no genres listed)":
                genre_counts[g] = genre_counts.get(g, 0) + 1

    df_genres = (
        pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"])
        .sort_values(by="Count", ascending=True)
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_genres["Count"],
                y=df_genres["Genre"],
                orientation="h",
                marker=dict(
                    color=df_genres["Count"],
                    colorscale="Viridis",
                    showscale=False,
                ),
                text=[f"{v:,}" for v in df_genres["Count"]],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="<b>Movie Catalog Distribution by Genre</b>",
        xaxis_title="Number of Movies",
        yaxis_title="Genre",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=520,
    )
    return fig


def create_evaluation_comparison_chart(summary_df: pd.DataFrame, metric_type: str = "P@") -> go.Figure:
    """Create grouped bar chart comparing algorithms across K values."""
    cols = [c for c in summary_df.columns if c.startswith(metric_type)]
    if not cols:
        cols = [c for c in summary_df.columns if c != "Algorithm"]

    fig = go.Figure()

    colors = {
        "Content-Based": "#3b82f6",
        "Collaborative": "#10b981",
        "Hybrid": "#a855f7",
    }

    for _, row in summary_df.iterrows():
        algo = row["Algorithm"]
        fig.add_trace(
            go.Bar(
                x=cols,
                y=[row[c] for c in cols],
                name=algo,
                marker_color=colors.get(algo, "#f59e0b"),
                text=[f"{row[c]:.3f}" for c in cols],
                textposition="auto",
            )
        )

    fig.update_layout(
        title=f"<b>Recommendation Algorithm Comparison ({metric_type}K)</b>",
        xaxis_title="Evaluation Cutoff (K)",
        yaxis_title=f"Average Score",
        barmode="group",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_weight_balance_chart(content_weight: float, collab_weight: float) -> go.Figure:
    """Create horizontal bar visualization for current hybrid agent weights."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Algorithm Balance"],
            x=[content_weight * 100],
            name=f"Content-Based ({int(content_weight * 100)}%)",
            orientation="h",
            marker=dict(color="#3b82f6"),
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Algorithm Balance"],
            x=[collab_weight * 100],
            name=f"Collaborative ({int(collab_weight * 100)}%)",
            orientation="h",
            marker=dict(color="#10b981"),
        )
    )

    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=10, r=10, t=25, b=25),
        template="plotly_dark",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
