"""
utils/data_processor.py — Load, clean, profile uploaded datasets
Adds: automatic plain-English insights (outliers, skew, correlations, cardinality)
"""

import pandas as pd
import numpy as np
import os


class DataProcessor:
    def __init__(self):
        self.last_df = None
        self.last_filepath = None

    # ─────────────────────────────────────────────
    def load_and_profile(self, filepath: str) -> dict:
        try:
            df = self._load(filepath)
        except Exception as e:
            return {"error": f"Could not read file: {str(e)}"}

        if df.shape[0] == 0 or df.shape[1] == 0:
            return {"error": "The uploaded file appears to be empty."}

        df = self._clean(df)
        self.last_df = df
        self.last_filepath = filepath

        missing = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        }

        column_types = {col: self._classify_column(df[col]) for col in df.columns}

        try:
            desc = df.describe(include="all").round(2)
            stats_html = desc.to_html(classes="stats-table", border=0)
        except Exception:
            stats_html = "<p>Statistics unavailable.</p>"

        preview_html = df.head(8).to_html(classes="data-table", border=0, index=False)

        insights = self._generate_insights(df, column_types, missing)

        return {
            "shape":        list(df.shape),
            "columns":      df.columns.tolist(),
            "column_types": column_types,
            "preview_html": preview_html,
            "stats_html":   stats_html,
            "missing":      missing,
            "insights":     insights,
        }

    # ─────────────────────────────────────────────
    def _load(self, filepath: str) -> pd.DataFrame:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        try:
            return pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding="latin-1")
        except pd.errors.ParserError:
            return pd.read_csv(filepath, encoding="utf-8", sep=None, engine="python")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip() for c in df.columns]

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan, "NULL": np.nan})

        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notna().sum() > 0.5 * len(df):
                        df[col] = converted
                except Exception:
                    pass

        return df

    def _classify_column(self, series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if series.nunique(dropna=True) < 20:
            return "categorical"
        return "text"

    # ─────────────────────────────────────────────
    # AUTO-INSIGHTS — plain-English data observations
    # ─────────────────────────────────────────────
    def _generate_insights(self, df, column_types, missing) -> list:
        insights = []
        n_rows = len(df)

        # Missing data
        if missing:
            worst_col = max(missing, key=missing.get)
            pct = round(missing[worst_col] / n_rows * 100, 1)
            insights.append({
                "icon": "fa-triangle-exclamation",
                "tone": "warn",
                "text": f"\"{worst_col}\" has {pct}% missing values — consider imputation or dropping it."
            })
        else:
            insights.append({
                "icon": "fa-circle-check",
                "tone": "good",
                "text": "No missing values detected across any column. Clean dataset."
            })

        # Numeric skew / outliers
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        for col in numeric_cols[:5]:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            skew = series.skew()
            if abs(skew) > 1.5:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "icon": "fa-chart-line",
                    "tone": "info",
                    "text": f"\"{col}\" is heavily {direction}-skewed (skew={round(skew,2)}) — a log transform may help linear models."
                })
                break

        # High-cardinality categoricals
        cat_cols = [c for c, t in column_types.items() if t == "categorical"]
        for col in cat_cols:
            nunique = df[col].nunique()
            if nunique > 15:
                insights.append({
                    "icon": "fa-layer-group",
                    "tone": "info",
                    "text": f"\"{col}\" has {nunique} unique categories — high cardinality may need grouping."
                })
                break

        # Strong correlations
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr().abs()
                np.fill_diagonal(corr.values, 0)
                max_val = corr.values.max()
                if max_val > 0.8:
                    idx = np.unravel_index(corr.values.argmax(), corr.shape)
                    c1, c2 = corr.index[idx[0]], corr.columns[idx[1]]
                    insights.append({
                        "icon": "fa-link",
                        "tone": "info",
                        "text": f"\"{c1}\" and \"{c2}\" are strongly correlated ({round(max_val,2)}) — possible redundancy."
                    })
            except Exception:
                pass

        # Dataset size note
        if n_rows < 100:
            insights.append({
                "icon": "fa-flask",
                "tone": "warn",
                "text": f"Only {n_rows} rows — model scores may be unstable with so little data."
            })
        elif n_rows > 50000:
            insights.append({
                "icon": "fa-server",
                "tone": "info",
                "text": f"Large dataset ({n_rows:,} rows) — consider sampling for faster iteration."
            })

        return insights[:5]
