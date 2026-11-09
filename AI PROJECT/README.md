# Smart Recommendation Agent Using Collaborative and Content-Based Filtering

An intelligent, multi-paradigm movie recommendation system combining **Content-Based Filtering** (TF-IDF vectorization and Cosine Similarity) and **Collaborative Filtering** (Item-Based rating pattern correlation) with **dynamic score weighting**, **cold-start handling**, and an **interactive Streamlit web dashboard**.

Built and evaluated on the benchmark **MovieLens (ml-latest-small)** dataset from GroupLens Research.

---

## 📌 1. Project Overview

Modern digital entertainment platforms host tens of thousands of catalog items. Users struggle with choice overload, while single-paradigm recommendation algorithms suffer from inherent structural flaws:
- **Content-Based Filtering** isolates items by feature similarity, failing to discover unexpected genres (lack of serendipity) and ignoring collective community rating wisdom.
- **Collaborative Filtering** depends exclusively on user-interaction matrices, failing catastrophically on new users and unrated items (**cold-start problem**) and suffering under high matrix sparsity (>98%).

This project constructs a **Smart Recommendation Agent** that unifies both approaches. By normalizing disparate score distributions and applying configurable weighted fusion, the agent delivers recommendations that are semantically relevant, supported by collective community trends, robust against cold-start conditions, and completely transparent with human-readable explanations.

---

## 🎯 2. Objectives & Key Features

- **Genuine Dual-Engine Algorithms**:
  - **Content-Based Engine**: Builds TF-IDF matrices over genre metadata and user-curated tags, constructing rating-weighted user interest vectors with Cosine Similarity.
  - **Collaborative Engine**: Constructs a sparse user-item rating matrix and applies Item-Based Adjusted Cosine (Pearson) similarity for rating predictions.
- **Hybrid Intelligent Agent**:
  - Dynamically combines normalized recommendation scores:
    $$\text{Hybrid Score} = (w_{\text{content}} \times \text{Score}_{\text{content}}) + (w_{\text{collab}} \times \text{Score}_{\text{collab}})$$
  - Validates and normalizes weights in real time ($w_1 + w_2 = 1.0$).
  - Seamless fallback for cold-start users and items with sparse history.
  - Excludes previously rated movies to ensure novel recommendations.
- **Explainability**:
  - Every recommendation includes an explicit explanation of *why* the title was suggested (identifying matching genres, favorite baseline movies, and community rating correlations).
- **Interactive Streamlit Web Dashboard**:
  - **Home Dashboard**: System summary, real-time KPI metrics, Bayesian top-rated and popular movies.
  - **Smart Agent**: Interactive user profile selector, cold-start preference simulator, real-time weight sliders, and multi-tab results with side-by-side model comparison.
  - **Standalone Explorers**: Dedicated tools to inspect content similarity and collaborative rating predictions.
  - **Dataset Analytics**: Interactive Plotly histograms, genre distribution bar charts, and a searchable catalog with filters.
  - **Offline Evaluation**: Benchmark runner computing Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, and MAE.
  - **About Section**: Comprehensive architectural and mathematical review.
- **Empirical Offline Evaluation**:
  - Per-user 80/20 train-test split without data leakage.
  - Standard sampled ranking protocol comparing Content-Based, Collaborative, and Hybrid approaches.
- **Automated Test Suite**:
  - 24 automated unit tests covering data loading, cleaning, vectorization, collaborative matrices, hybrid weighting, and metrics.

---

## 🏗️ 3. System Architecture

```
+-------------------------------------------------------------------------+
|                        Streamlit Web Interface                          |
|   [Home]   [Smart Agent]   [Content-Based]   [Collaborative]   [Eval]   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     Hybrid Recommendation Agent                         |
|   - Dynamic Weight Normalization: w_cb + w_cf = 1.0                     |
|   - Score Normalization: Min-Max Scaling to [0.0, 1.0]                  |
|   - Fallback Strategy: Cold-Start Graceful Degradation                  |
|   - Synthesized Explainability Generator                                |
+--------------------+-------------------------------+--------------------+
                     |                               |
                     v                               v
+------------------------------------+   +------------------------------------+
|       Content-Based Engine         |   |   Collaborative Filtering Engine   |
|   - Metadata Soup (Genres + Tags)  |   |   - Sparse User-Item Matrix (CSR)  |
|   - TF-IDF Vectorizer (ngram 1-2)  |   |   - Item-Item Adjusted Cosine (r-u)|
|   - User Preference Profile Vector |   |   - Vectorized Rating Prediction   |
|   - Cosine Similarity Matching     |   |   - Top Contributing Neighbors     |
+------------------------------------+   +------------------------------------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                          MovieLens Data Layer                           |
|   - Dataset: ml-latest-small (9,742 movies, 100,836 ratings, 610 users) |
|   - Auto-downloader with SSL fallback, zip extraction, and disk cache   |
|   - Data cleaning, year extraction, metadata enrichment, deduplication  |
+-------------------------------------------------------------------------+
```

---

## 📐 4. Algorithm Details & Mathematical Formulations

### 4.1 Content-Based Filtering
1. **Metadata Soup Construction**: Each movie's genres (e.g. `Action|Adventure|Sci-Fi`) and all user-contributed tags are tokenized into a unified text descriptor. Genre tokens are given balanced weight relative to unstructured tags.
2. **TF-IDF Vectorization**:
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \left( \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1 \right)$$
3. **User Taste Profile**:
   For a user $u$ who rated movies $M_u^+$, a weighted average interest vector is computed:
   $$\mathbf{p}_u = \sum_{m \in M_u^+} w_m \cdot \mathbf{v}_m, \quad \text{where } w_m = \frac{r_{u, m} - r_{\min} + 0.5}{r_{\max} - r_{\min} + 0.5}$$
   $$\mathbf{p}_u \leftarrow \frac{\mathbf{p}_u}{\|\mathbf{p}_u\|}$$
4. **Cosine Similarity**:
   $$\text{Score}_{\text{content}}(u, i) = \frac{\mathbf{p}_u \cdot \mathbf{v}_i}{\|\mathbf{p}_u\| \|\mathbf{v}_i\|}$$

### 4.2 Collaborative Filtering (Item-Based)
1. **Mean-Centered Ratings (Adjusted Cosine)**:
   Accounts for user rating scale discrepancies (e.g. strict raters vs. lenient raters):
   $$\text{Sim}(i, j) = \frac{\sum_{u \in U} (r_{u, i} - \bar{r}_u)(r_{u, j} - \bar{r}_u)}{\sqrt{\sum_{u \in U} (r_{u, i} - \bar{r}_u)^2} \sqrt{\sum_{u \in U} (r_{u, j} - \bar{r}_u)^2}}$$
2. **Vectorized Rating Prediction**:
   $$\hat{r}_{u, i} = \frac{\sum_{j \in \text{Rated}(u)} \max(\text{Sim}(i, j), 0) \cdot r_{u, j}}{\sum_{j \in \text{Rated}(u)} \max(\text{Sim}(i, j), 0)}$$
   Predictions are bounded to the valid range $[0.5, 5.0]$ and normalized to $[0.0, 1.0]$.

### 4.3 Hybrid Score Fusion
1. **Min-Max Score Normalization**:
   $$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
2. **Linear Weighted Combination**:
   $$\text{Score}_{\text{hybrid}} = w_{\text{content}} \cdot S_{\text{content}}^{\text{norm}} + w_{\text{collab}} \cdot S_{\text{collab}}^{\text{norm}}$$
   Subject to: $w_{\text{content}} \ge 0$, $w_{\text{collab}} \ge 0$, $w_{\text{content}} + w_{\text{collab}} = 1.0$.

---

## 📊 5. Empirical Evaluation & Benchmark Results

The system includes an offline evaluation module that splits user ratings into 80% train and 20% test sets per user. Test ratings $\ge 3.5$ are treated as relevant ground-truth positives.

Evaluation protocol: Sampled ranking protocol ranking relevant test items among 100 randomly sampled negative catalog items (standard academic protocol).

### Measured Offline Results (MovieLens ml-latest-small)

| Algorithm | Precision@5 | Precision@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Hit Rate@10 | MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Content-Based** | 0.170 | 0.140 | 0.067 | 0.132 | 0.152 | 0.154 | 0.550 | 0.219 |
| **Collaborative** | 0.260 | 0.290 | 0.115 | 0.268 | 0.246 | 0.304 | 0.800 | 0.379 |
| **Hybrid Agent** | **0.310** | **0.300** | **0.163** | **0.299** | **0.315** | **0.347** | **0.900** | **0.477** |

- **Collaborative Filtering Rating Error**:
  - **RMSE (Root Mean Squared Error)**: `0.9123`
  - **MAE (Mean Absolute Error)**: `0.6989`

### Key Findings
1. **Hybrid Outperforms Single Models**: The Hybrid Agent achieves higher Precision@5 (0.310 vs 0.260 / 0.170) and NDCG@10 (0.347 vs 0.304 / 0.154), demonstrating that item metadata and collective rating signals complement each other.
2. **Cold-Start Resilience**: When collaborative matrix data is scarce or nonexistent, the hybrid engine smoothly shifts weight to content similarity without runtime failure.

---

## 📁 6. Project Structure

```
smart-recommendation-agent/
│
├── app.py                      # Main interactive Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & project review guide
├── .gitignore                  # Git ignore rules
│
├── data/                       # MovieLens dataset cache
│   └── ml-latest-small/
│       ├── movies.csv          # Movie IDs, titles, and genres
│       ├── ratings.csv         # User ratings and timestamps
│       ├── tags.csv            # User-assigned tags
│       └── links.csv           # IMDb and TMDb identifiers
│
├── src/                        # Core recommendation modules
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, cleaning, statistics, & Bayesian rating
│   ├── content_based.py        # TF-IDF vectorization & Cosine Similarity engine
│   ├── collaborative.py        # Item-Based CF, Pearson correlation & vectorized prediction
│   ├── hybrid.py               # Hybrid Agent, normalization, dynamic fusion & fallback
│   ├── evaluation.py           # Offline evaluation, train/test split, Precision/Recall/NDCG/RMSE
│   └── utils.py                # Visual badges, styling, and Plotly chart renderers
│
└── tests/                      # Automated test suite
    ├── test_data_loader.py     # Data loader & cleaning unit tests
    ├── test_content_based.py   # Content-based engine unit tests
    ├── test_collaborative.py   # Collaborative filtering unit tests
    ├── test_hybrid.py          # Hybrid agent & weight validation unit tests
    └── test_evaluation.py      # Offline evaluation metrics unit tests
```

---

## 💻 7. Installation & Setup Guide (Windows / VS Code)

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your Windows machine.
- PowerShell or Command Prompt.

### Step-by-Step Instructions

#### Step 1: Open Terminal in Project Directory
Open VS Code in the project root:
```powershell
cd "c:\Users\ISHITHA\OneDrive\Desktop\AI PROJECT"
```

#### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv .venv
```

#### Step 3: Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```
*(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### Step 4: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

#### Step 5: Run Automated Tests
Execute the pytest suite to verify all components:
```powershell
python -m pytest tests/ -v
```
*(All 24 unit tests will execute and pass).*

#### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Or with python module invocation:
```powershell
python -m streamlit run app.py
```

The application will open automatically in your browser at:
`http://localhost:8501`

---

## 📥 8. Dataset Information & Manual Download Instructions

The application automatically downloads and extracts the official **MovieLens ml-latest-small** dataset on initial launch.

If you are working offline or automatic download is prevented by firewall restrictions, you can load the dataset manually:
1. Download `ml-latest-small.zip` directly from GroupLens:
   https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
2. Extract the archive into the `data/` folder inside the project directory:
   `c:\Users\ISHITHA\OneDrive\Desktop\AI PROJECT\data\ml-latest-small\`
3. Verify that the files exist at:
   - `data/ml-latest-small/movies.csv`
   - `data/ml-latest-small/ratings.csv`
   - `data/ml-latest-small/tags.csv`

The data loader detects existing files and uses them immediately without re-downloading.

---

## 🛠️ 9. Technology Stack Summary

| Technology | Purpose |
|---|---|
| **Python** | Backend algorithms, matrix processing, and agent logic |
| **Streamlit** | Interactive, responsive web interface |
| **Pandas & NumPy** | Data ingestion, vector manipulations, and matrix indexing |
| **Scikit-learn** | TF-IDF vectorization and cosine similarity computation |
| **SciPy** | Sparse matrix storage (CSR format) |
| **Plotly** | Interactive data visualizations and metric charts |
| **Pytest** | Comprehensive automated test coverage |
| **GroupLens MovieLens** | Real-world benchmark movie ratings dataset |

---

## 🔮 10. Limitations & Future Scope

### Current Limitations
1. **Static Metadata**: Movie metadata relies on catalog titles, genres, and user tags; rich synopsis embeddings (e.g. BERT/LLM summaries) could enhance fine-grained theme matching.
2. **Offline Evaluation Bias**: Offline testing only observes historical ratings; unobserved items recommended by the agent are marked as negatives in offline recall even if the user would genuinely enjoy them.

### Future Scope
1. **Deep Learning Embeddings**: Integrate Sentence-Transformers or OpenAI/Gemini embeddings to encode movie plot summaries.
2. **Neural Collaborative Filtering (NCF)**: Replace matrix cosine similarity with neural collaborative architectures to model non-linear user-item interactions.
3. **Session-Based Recurrent Modeling**: Capture real-time sequential user watch intent using Transformer/GRU architectures.
