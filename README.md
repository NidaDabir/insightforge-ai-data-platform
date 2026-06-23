# ⚡ InsightForge — AI-Powered Data Intelligence Platform

> Upload CSV/Excel → Auto-profile with insights → Train & compare 3 ML models live → Explain with SHAP → REST API

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
| Deploy | **Gunicorn** + Render/Railway/Heroku |
| Explainability | **SHAP** (optional) |

---

## 📁 Project Structure

```
insightforge/
├── app.py                  # Flask app — routes + JSON train endpoint
├── requirements.txt
├── Procfile
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

## ⚙️ Local Setup

```bash
git clone https://github.com/yourusername/insightforge.git
cd insightforge
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:5050
```

For SHAP explainability plots, also run: `pip install shap`

---

## 🚢 Deploy to Render (Free)

1. Push to GitHub
2. [render.com](https://render.com) → New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

> SQLite resets on free-tier redeploys. Swap in PostgreSQL for persistence in production.

---

## 🎯 Why This Stands Out on a Resume

- ✅ **Full-stack Flask app** with real async UX (fetch API, not just forms)
- ✅ **Multi-model auto-ML** — not a single hardcoded classifier
- ✅ **Explainable AI** — feature importance everywhere, SHAP when available
- ✅ **Production REST API** — 5 documented endpoints
- ✅ **Per-user persistence** via SQLite, full project/run history
- ✅ **Deployment-ready** — Gunicorn + Procfile
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
