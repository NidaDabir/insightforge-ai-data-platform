"""
InsightForge — AI-Powered Data Intelligence Platform
Flask backend: auth, upload, AJAX model training, REST API, project history.
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from functools import wraps
import os
import json
import uuid

from utils.data_processor import DataProcessor
from utils.ml_engine import MLEngine
from utils.visualizer import Visualizer
from utils.db import Database

app = Flask(__name__)
app.secret_key = "insightforge_secret_2024_v3"

UPLOAD_FOLDER = os.path.join("static", "uploads")
PLOT_FOLDER   = os.path.join("static", "plots")
for folder in [UPLOAD_FOLDER, PLOT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

db  = Database()
dp  = DataProcessor()
ml  = MLEngine()
viz = Visualizer()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif db.get_user_by_email(email):
            error = "An account with that email already exists."
        else:
            user_id = db.create_user(username, email, password)
            session["user_id"]  = user_id
            session["username"] = username
            return redirect(url_for("dashboard"))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.authenticate_user(email, password)
        if user:
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ══════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    projects = db.get_projects(user_id)
    stats = db.get_user_stats(user_id)
    return render_template("dashboard.html", username=session["username"],
                           projects=projects, stats=stats)


# ══════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        return render_template("error.html", message="Unsupported file type. Please upload CSV or Excel.")

    unique_fn = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_fn)
    file.save(filepath)

    result = dp.load_and_profile(filepath)
    if "error" in result:
        return render_template("error.html", message=result["error"])

    project_id = db.create_project(
        user_id=session["user_id"], name=file.filename, filepath=filepath,
        shape=result["shape"], column_types=result["column_types"]
    )

    return render_template(
        "analysis.html",
        username=session["username"], project_id=project_id, filename=file.filename,
        shape=result["shape"], columns=result["columns"], col_types=result["column_types"],
        preview=result["preview_html"], stats_html=result["stats_html"],
        missing=result["missing"], insights=result["insights"], filepath=filepath,
    )


# ══════════════════════════════════════════════
# TRAIN — JSON API consumed by frontend JS (no page reload)
# ══════════════════════════════════════════════
@app.route("/train", methods=["POST"])
@login_required
def train():
    try:
        filepath   = request.form.get("filepath")
        project_id = request.form.get("project_id")
        target     = request.form.get("target")
        features   = request.form.getlist("features")

        if not filepath or not target or not features:
            return jsonify({"error": "Missing filepath, target, or features."}), 400

        result = dp.load_and_profile(filepath)
        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        df = dp.last_df
        train_result = ml.train(df, features, target)

        plot_urls = viz.generate_all(df, features, target, PLOT_FOLDER)

        shap_url = None
        if train_result.get("supports_shap") and train_result.get("X_test") is not None:
            shap_url = viz.generate_shap(train_result["model"], train_result["X_test"], features, PLOT_FOLDER)

        conf_url = None
        if train_result.get("conf_matrix") is not None:
            conf_url = viz.plot_confusion_matrix(
                train_result["conf_matrix"], train_result.get("classes", []), PLOT_FOLDER
            )

        db.save_model_run(
            project_id=project_id, target=target, features=features,
            model_type=train_result["model_type"], task=train_result["task"],
            score=train_result["score"], metric=train_result["metric"],
            all_scores=train_result.get("all_scores"),
        )

        return jsonify({
            "success":           True,
            "accuracy":          train_result["score"],
            "metric":            train_result["metric"],
            "model_type":        train_result["model_type"],
            "task":              train_result["task"],
            "feature_importance": train_result.get("feature_importance"),
            "all_scores":        train_result.get("all_scores"),
            "extra_metrics":     train_result.get("extra_metrics"),
            "plot_urls":         plot_urls,
            "shap_url":          shap_url,
            "conf_url":          conf_url,
            "classes":           train_result.get("classes"),
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


# ══════════════════════════════════════════════
# PROJECT HISTORY
# ══════════════════════════════════════════════
@app.route("/history")
@login_required
def history():
    projects = db.get_projects(session["user_id"])
    return render_template("history.html", username=session["username"], projects=projects)


@app.route("/project/<int:project_id>")
@login_required
def project_detail(project_id):
    project = db.get_project(project_id, session["user_id"])
    if not project:
        return redirect(url_for("history"))
    runs = db.get_model_runs(project_id)
    return render_template("project_detail.html", username=session["username"],
                           project=project, runs=runs)


@app.route("/project/<int:project_id>/delete", methods=["POST"])
@login_required
def delete_project(project_id):
    db.delete_project(project_id, session["user_id"])
    return redirect(url_for("history"))


@app.route("/download/<int:project_id>")
@login_required
def download_results(project_id):
    project = db.get_project(project_id, session["user_id"])
    if not project:
        return "Not found", 404
    runs = db.get_model_runs(project_id)
    data = {
        "project": {"name": project["name"], "shape": project["shape"], "created": project["created_at"]},
        "model_runs": runs
    }
    tmp_path = os.path.join("static", f"results_{project_id}.json")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return send_file(tmp_path, as_attachment=True, download_name=f"insightforge_results_{project_id}.json")


# ══════════════════════════════════════════════
# REST API — v1
# ══════════════════════════════════════════════
@app.route("/api/v1/status", methods=["GET"])
def api_status():
    return jsonify({"status": "ok", "version": "2.0.0", "service": "InsightForge API"})


@app.route("/api/v1/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f"api_{uuid.uuid4().hex}{ext}")
    file.save(tmp_path)

    result = dp.load_and_profile(tmp_path)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({
        "filepath": tmp_path, "shape": result["shape"], "columns": result["columns"],
        "types": result["column_types"], "missing": result["missing"]
    })


@app.route("/api/v1/train", methods=["POST"])
def api_train():
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    target   = data.get("target")
    features = data.get("features", [])

    if not filepath or not target or not features:
        return jsonify({"error": "filepath, target, features required"}), 400

    result = dp.load_and_profile(filepath)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    df = dp.last_df
    try:
        train_result = ml.train(df, features, target)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    return jsonify({
        "model_type": train_result["model_type"], "task": train_result["task"],
        "score": train_result["score"], "metric": train_result["metric"],
        "feature_importance": train_result.get("feature_importance"),
        "all_scores": train_result.get("all_scores"),
    })


@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    target   = data.get("target")
    features = data.get("features", [])
    rows     = data.get("rows", [])

    if not filepath or not target or not features or not rows:
        return jsonify({"error": "filepath, target, features, rows required"}), 400

    result = dp.load_and_profile(filepath)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    df = dp.last_df
    preds = ml.predict(df, features, target, rows)
    return jsonify({"predictions": preds})


@app.route("/api/v1/projects", methods=["GET"])
@login_required
def api_projects():
    projects = db.get_projects(session["user_id"])
    return jsonify({"projects": projects})


# ══════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message="Internal server error."), 500


if __name__ == "__main__":
    db.init_db()
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False)
