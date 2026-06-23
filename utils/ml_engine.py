"""
utils/ml_engine.py — Multi-model auto-ML engine
Trains several candidate models, reports all scores, returns the best.
Handles: classification, regression, NLP text classification.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, precision_score, recall_score, f1_score
)


class MLEngine:

    def train(self, df: pd.DataFrame, features: list, target: str) -> dict:
        if target in features:
            features = [f for f in features if f != target]
        if not features:
            raise ValueError("Select at least one feature column.")

        df = df.copy().dropna(subset=[target])
        if len(df) < 10:
            raise ValueError("Not enough rows with a valid target to train (need at least 10).")

        y = df[target]

        text_features = [f for f in features if df[f].dtype == object and df[f].nunique() > 20]
        if text_features:
            return self._train_nlp(df, text_features[0], target, y)

        return self._train_tabular(df, features, target, y)

    # ──────────────────────────────────────────────────────────
    # NLP CLASSIFICATION
    # ──────────────────────────────────────────────────────────
    def _train_nlp(self, df, text_col, target, y):
        X_text = df[text_col].fillna("").astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if len(set(y_enc)) > 1 and min(np.bincount(y_enc)) >= 2 else None
        )

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))),
            ("clf",   LogisticRegression(max_iter=500, C=1.0))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)

        classes = le.classes_.tolist()
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "model":              pipe,
            "model_type":         "Logistic Regression (TF-IDF)",
            "task":               "nlp_classification",
            "score":              acc,
            "metric":             "Accuracy %",
            "feature_importance": None,
            "conf_matrix":        cm,
            "classes":            [str(c) for c in classes],
            "X_test":             None,
            "supports_shap":      False,
            "all_scores":         [{"model": "Logistic Regression (TF-IDF)", "score": acc}],
            "extra_metrics":      {},
        }

    # ──────────────────────────────────────────────────────────
    # TABULAR ROUTING
    # ──────────────────────────────────────────────────────────
    def _train_tabular(self, df, features, target, y):
        X = df[features].copy()

        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].map(lambda v: v.toordinal() if pd.notnull(v) else 0)

        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median(numeric_only=True))
        X = X.loc[:, ~X.columns.duplicated()]

        is_classification = (
            y.dtype == object or
            str(y.dtype).startswith("bool") or
            y.nunique() <= 20
        )

        stratify = None
        if is_classification:
            vc = y.value_counts()
            if vc.min() >= 2 and len(vc) > 1:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        if is_classification:
            return self._classification(X_train, X_test, y_train, y_test, X.columns.tolist())
        else:
            return self._regression(X_train, X_test, y_train, y_test, X.columns.tolist())

    # ──────────────────────────────────────────────────────────
    # CLASSIFICATION — multi-model comparison
    # ──────────────────────────────────────────────────────────
    def _classification(self, X_train, X_test, y_train, y_test, all_cols):
        candidates = {
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.1, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Logistic Regression": LogisticRegression(max_iter=1000),
        }

        results = {}
        fitted = {}
        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results[name] = acc
                fitted[name] = (model, pred)
            except Exception:
                continue

        if not results:
            raise ValueError("All candidate models failed to train on this data.")

        best_name = max(results, key=results.get)
        best_model, best_pred = fitted[best_name]

        acc = round(results[best_name] * 100, 2)
        classes = sorted(y_test.unique().tolist())
        cm = confusion_matrix(y_test, best_pred, labels=classes).tolist()

        try:
            precision = round(precision_score(y_test, best_pred, average="weighted", zero_division=0) * 100, 2)
            recall    = round(recall_score(y_test, best_pred, average="weighted", zero_division=0) * 100, 2)
            f1        = round(f1_score(y_test, best_pred, average="weighted", zero_division=0) * 100, 2)
        except Exception:
            precision = recall = f1 = None

        feat_imp = self._feature_importance(best_model, all_cols)

        all_scores = [{"model": k, "score": round(v * 100, 2)} for k, v in
                      sorted(results.items(), key=lambda x: -x[1])]

        return {
            "model":              best_model,
            "model_type":         best_name,
            "task":               "classification",
            "score":              acc,
            "metric":             "Accuracy %",
            "feature_importance": feat_imp,
            "conf_matrix":        cm,
            "classes":            [str(c) for c in classes],
            "X_test":             X_test,
            "supports_shap":      best_name in ("Gradient Boosting", "Random Forest"),
            "all_scores":         all_scores,
            "extra_metrics": {
                "Precision": precision, "Recall": recall, "F1 Score": f1
            },
        }

    # ──────────────────────────────────────────────────────────
    # REGRESSION — multi-model comparison
    # ──────────────────────────────────────────────────────────
    def _regression(self, X_train, X_test, y_train, y_test, all_cols):
        candidates = {
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=120, max_depth=4, learning_rate=0.1, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=120, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Linear Regression": LinearRegression(),
        }

        results = {}
        fitted = {}
        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)
                results[name] = r2
                fitted[name] = (model, pred)
            except Exception:
                continue

        if not results:
            raise ValueError("All candidate models failed to train on this data.")

        best_name = max(results, key=results.get)
        best_model, best_pred = fitted[best_name]

        r2 = round(results[best_name] * 100, 2)
        mae = round(mean_absolute_error(y_test, best_pred), 4)
        rmse = round(mean_squared_error(y_test, best_pred) ** 0.5, 4)

        feat_imp = self._feature_importance(best_model, all_cols)

        all_scores = [{"model": k, "score": round(v * 100, 2)} for k, v in
                      sorted(results.items(), key=lambda x: -x[1])]

        return {
            "model":              best_model,
            "model_type":         best_name,
            "task":               "regression",
            "score":              r2,
            "metric":             "R² Score %",
            "feature_importance": feat_imp,
            "conf_matrix":        None,
            "classes":            [],
            "X_test":             X_test,
            "supports_shap":      best_name in ("Gradient Boosting", "Random Forest"),
            "all_scores":         all_scores,
            "extra_metrics": {
                "MAE": mae, "RMSE": rmse
            },
        }

    # ──────────────────────────────────────────────────────────
    # PREDICT — for REST API
    # ──────────────────────────────────────────────────────────
    def predict(self, df, features, target, rows):
        result = self.train(df, features, target)
        model = result["model"]

        new_df = pd.DataFrame(rows)
        for col in new_df.columns:
            if pd.api.types.is_datetime64_any_dtype(new_df[col]):
                new_df[col] = new_df[col].map(lambda v: v.toordinal() if pd.notnull(v) else 0)
        new_df = pd.get_dummies(new_df, drop_first=True)

        if hasattr(model, "n_features_in_"):
            n = model.n_features_in_
            new_df = new_df.reindex(columns=range(n), fill_value=0)

        try:
            preds = model.predict(new_df).tolist()
        except Exception as e:
            preds = [f"Error: {str(e)}"]
        return preds

    # ──────────────────────────────────────────────────────────
    def _feature_importance(self, model, col_names):
        try:
            imp = model.feature_importances_
            paired = sorted(zip(col_names, imp.tolist()), key=lambda x: x[1], reverse=True)[:15]
            return [{"feature": k, "importance": round(v, 4)} for k, v in paired]
        except AttributeError:
            try:
                imp = np.abs(model.coef_).flatten()
                paired = sorted(zip(col_names, imp.tolist()), key=lambda x: x[1], reverse=True)[:15]
                total = sum(v for _, v in paired) or 1
                return [{"feature": k, "importance": round(v / total, 4)} for k, v in paired]
            except Exception:
                return None
