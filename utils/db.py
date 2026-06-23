"""
utils/db.py — SQLite persistence layer
Tables: users, projects, model_runs
"""

import sqlite3
import hashlib
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "insightforge.db")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class Database:
    def __init__(self):
        self.db_path = os.path.abspath(DB_PATH)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────── INIT ───────────────
    def init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT    NOT NULL,
                    email      TEXT    NOT NULL UNIQUE,
                    password   TEXT    NOT NULL,
                    created_at REAL    DEFAULT (strftime('%s','now'))
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      INTEGER NOT NULL,
                    name         TEXT    NOT NULL,
                    filepath     TEXT,
                    shape        TEXT,
                    column_types TEXT,
                    created_at   REAL    DEFAULT (strftime('%s','now')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id   INTEGER NOT NULL,
                    target       TEXT,
                    features     TEXT,
                    model_type   TEXT,
                    task         TEXT,
                    score        REAL,
                    metric       TEXT,
                    all_scores   TEXT,
                    ran_at       REAL DEFAULT (strftime('%s','now')),
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                );
            """)

    # ─────────────── USERS ───────────────
    def create_user(self, username, email, password) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                (username, email, _hash(password))
            )
            return cur.lastrowid

    def get_user_by_email(self, email):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            return dict(row) if row else None

    def authenticate_user(self, email, password):
        user = self.get_user_by_email(email)
        if user and user["password"] == _hash(password):
            return user
        return None

    # ─────────────── PROJECTS ───────────────
    def create_project(self, user_id, name, filepath, shape, column_types) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO projects (user_id, name, filepath, shape, column_types)
                   VALUES (?,?,?,?,?)""",
                (user_id, name, filepath, json.dumps(shape), json.dumps(column_types))
            )
            return cur.lastrowid

    def get_projects(self, user_id: int):
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT p.*, COUNT(m.id) as run_count, MAX(m.score) as best_score
                   FROM projects p
                   LEFT JOIN model_runs m ON m.project_id = p.id
                   WHERE p.user_id=?
                   GROUP BY p.id
                   ORDER BY p.created_at DESC""",
                (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_project(self, project_id: int, user_id: int):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id=? AND user_id=?",
                (project_id, user_id)
            ).fetchone()
            return dict(row) if row else None

    def get_project_by_id_only(self, project_id: int):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            return dict(row) if row else None

    def delete_project(self, project_id: int, user_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM model_runs WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id=? AND user_id=?", (project_id, user_id))

    def get_user_stats(self, user_id: int) -> dict:
        with self._conn() as conn:
            n_proj = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            n_runs = conn.execute(
                """SELECT COUNT(*) FROM model_runs m JOIN projects p ON p.id=m.project_id
                   WHERE p.user_id=?""", (user_id,)
            ).fetchone()[0]
            avg_score = conn.execute(
                """SELECT AVG(m.score) FROM model_runs m JOIN projects p ON p.id=m.project_id
                   WHERE p.user_id=?""", (user_id,)
            ).fetchone()[0]
            best_score = conn.execute(
                """SELECT MAX(m.score) FROM model_runs m JOIN projects p ON p.id=m.project_id
                   WHERE p.user_id=?""", (user_id,)
            ).fetchone()[0]
            return {
                "projects":   n_proj,
                "runs":       n_runs,
                "avg_score":  round(avg_score, 1) if avg_score else 0,
                "best_score": round(best_score, 1) if best_score else 0,
            }

    # ─────────────── MODEL RUNS ───────────────
    def save_model_run(self, project_id, target, features, model_type, task, score, metric, all_scores=None):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO model_runs
                   (project_id, target, features, model_type, task, score, metric, all_scores)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (project_id, target, json.dumps(features), model_type, task,
                 score, metric, json.dumps(all_scores) if all_scores else None)
            )

    def get_model_runs(self, project_id: int):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM model_runs WHERE project_id=? ORDER BY ran_at DESC",
                (project_id,)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for key in ("features", "all_scores"):
                    try:
                        d[key] = json.loads(d[key]) if d[key] else None
                    except Exception:
                        pass
                result.append(d)
            return result
