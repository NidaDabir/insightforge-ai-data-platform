# ⚡ InsightForge — AI-Powered Data Intelligence Platform

[![Live Demo](https://img.shields.io/badge/demo-live-34d9c5?style=for-the-badge)](https://insightforge-ai-data-platform.onrender.com)
[![Docker](https://img.shields.io/badge/containerized-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#-run-with-docker)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](#)

> Upload CSV/Excel → Auto-profile with insights → Train & compare 3 ML models live → Explain with SHAP → REST API

**[🚀 Live Demo](https://insightforge-ai-data-platform.onrender.com)** &nbsp;|&nbsp; **[📂 Source](https://github.com/NidaDabir/insightforge-ai-data-platform)**

> ⏳ Hosted on Render's free tier — the app spins down after inactivity, so the first load can take 30-50 seconds to wake up. Totally normal, just give it a moment.

---

## 🧠 What's New in v2

| v1 | v2 |
|----|----|
| Single model trained per run | **3 models trained & compared** automatically (GBM, Random Forest, Linear/Logistic) |
| Full page reload to see results | **AJAX training** — results and charts stream in live, no reload |
| Basic column list | **Auto-generated plain-English insights** (skew, correlation, missing data, cardinality) |
| Generic dark theme | **Custom violet/teal design system** with animated pipeline visual |
| Accuracy only | **Precision / Recall / F1 / MAE / RMSE** depending on task |

---

## 🧭 What It Does

| Step | What happens |
|------|--------------|
| **Upload** | CSV or Excel (.xlsx/.xls), drag-and-drop |
| **Profile** | Column types, missing values, descriptive stats, auto-insights |
| **Model** | Trains Gradient Boosting + Random Forest + Linear/Logistic baseline, reports the winner |
| **Explain** | Feature importance for every run; SHAP beeswarm plot when installed |
| **Visualize** | 4 auto-generated charts: correlation heatmap, distribution, scatter w/ trendline, category breakdown |
| **History** | Every project + every model run stored in SQLite, scoped per user |
| **API** | 5 REST endpoints for programmatic access |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | **Flask 3.0** — routing, sessions, JSON API |
| ML | **scikit-learn** — GradientBoosting, RandomForest, Logistic/Linear Regression, TF-IDF NLP |
| Data | **pandas**, **NumPy** |
| Visualization | **Matplotlib + Seaborn**, custom dark theme |
| Database | **SQLite** (stdlib `sqlite3`) |
| Auth | SHA-256 password hashing + Flask sessions |
| Frontend | Vanilla JS (fetch API), no framework — fully async training UI |
| Containerization | **Docker** + Docker Compose |
| Deploy | **Gunicorn** + Render (via Docker) |
| Explainability | **SHAP** (optional) |

---

## 📁 Project Structure

```
insightforge/
├── app.py                  # Flask app — routes + JSON train endpoint
├── Dockerfile               # Container image definition
├── docker-compose.yml       # One-command local orchestration
├── .dockerignore
├── requirements.txt
├── Procfile                 # Fallback for non-Docker PaaS deploys
├── utils/
│   ├── db.py               # SQLite layer (users, projects, model_runs)
│   ├── data_processor.py   # Load, clean, profile, auto-insights
│   ├── ml_engine.py        # Multi-model training & comparison
│   └── visualizer.py       # Chart generation (themed)
├── templates/
│   ├── base.html           # App shell, sidebar, toast system
│   ├── landing.html        # Marketing page w/ animated pipeline
│   ├── signup.html / login.html
│   ├── dashboard.html
│   ├── analysis.html       # AJAX-driven ML workspace (no reload)
│   ├── history.html
│   ├── project_detail.html
│   └── error.html
└── static/
    ├── uploads/
    └── plots/
```

---

## 🔌 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/status` | Health check |
| `POST` | `/api/v1/upload` | Upload CSV/Excel → column profile |
| `POST` | `/api/v1/train`  | Train & compare models → best score + importance |
| `POST` | `/api/v1/predict`| Predict on new rows |
| `GET`  | `/api/v1/projects`| List user's projects (auth required) |

### Example

```bash
curl -X POST http://localhost:5050/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "static/uploads/mydata.csv",
    "target": "SalePrice",
    "features": ["GrLivArea", "OverallQual", "YearBuilt"]
  }'
```

```json
{
  "model_type": "Gradient Boosting",
  "task": "regression",
  "score": 90.66,
  "metric": "R² Score %",
  "feature_importance": [{"feature": "OverallQual", "importance": 0.51}],
  "all_scores": [
    {"model": "Linear Regression", "score": 90.66},
    {"model": "Random Forest", "score": 88.16},
    {"model": "Gradient Boosting", "score": 87.55}
  ]
}
```

---

## 🐳 Run with Docker

The fastest way to run InsightForge — no Python setup, no venv, no dependency conflicts.

```bash
git clone https://github.com/NidaDabir/insightforge-ai-data-platform.git
cd insightforge-ai-data-platform

docker compose up --build
# → http://localhost:5050
```

That's it. Docker Compose builds the image, installs every dependency inside the container, and persists your SQLite data + uploaded files in named volumes so they survive restarts.

To stop it: `docker compose down`
To rebuild after a code change: `docker compose up --build`

**Or with plain Docker (no Compose):**

```bash
docker build -t insightforge .
docker run -p 5050:5050 insightforge
```

---

## ⚙️ Run Without Docker (local Python)

```bash
git clone https://github.com/NidaDabir/insightforge-ai-data-platform.git
cd insightforge-ai-data-platform
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:5050
```

For SHAP explainability plots, also run: `pip install shap`

---

## 🚢 Deployment

InsightForge is deployed as a **Docker container** on [Render](https://render.com).

**To deploy your own copy:**

1. Push this repo to your GitHub
2. [render.com](https://render.com) → **New** → **Web Service** → connect your repo
3. Render auto-detects the `Dockerfile` — choose **Docker** as the environment (no build/start commands needed, they're baked into the image)
4. Add environment variable: `SECRET_KEY` = (any random string)
5. Deploy

> SQLite resets on Render's free-tier redeploys since the filesystem isn't persistent across deploys. For a portfolio demo this is fine — for real production use, swap in PostgreSQL.

---

## 🎯 Why This Stands Out on a Resume

- ✅ **Containerized with Docker** — multi-stage-ready Dockerfile + docker-compose for one-command local setup
- ✅ **Full-stack Flask app** with real async UX (fetch API, not just forms)
- ✅ **Multi-model auto-ML** — not a single hardcoded classifier
- ✅ **Explainable AI** — feature importance everywhere, SHAP when available
- ✅ **Production REST API** — 5 documented endpoints
- ✅ **Per-user persistence** via SQLite, full project/run history
- ✅ **Live deployed demo** — not just code, a working product
- ✅ **Original UI system** — no Bootstrap, custom design tokens, animated brand element

---

## 🧩 Possible Extensions

- Swap SQLite → PostgreSQL via SQLAlchemy
- Add Celery/RQ for background training on large files
- Add PyTorch/TensorFlow deep learning path
- Add WebSocket progress streaming during training

---

## 📜 License

MIT — free to use for portfolio, learning, or building upon.
