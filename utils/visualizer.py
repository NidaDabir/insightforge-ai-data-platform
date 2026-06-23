"""
utils/visualizer.py — Generate Matplotlib/Seaborn charts as PNGs
Theme matched to the app's violet/cyan dark UI.
"""

import os
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

BG      = "#0b1220"
PANEL   = "#11192c"
GRID    = "#1f2b45"
FG      = "#cbd5e1"
ACCENT  = "#7c6cf5"
ACCENT2 = "#34d9c5"
ACCENT3 = "#f5a623"


def _new_fig(figsize=(8, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color("#f1f5f9")
    return fig, ax


def _save(fig, folder, prefix) -> str:
    fname = f"{prefix}_{int(time.time()*1000)}.png"
    path = os.path.join(folder, fname)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return f"/static/plots/{fname}"


class Visualizer:

    def generate_all(self, df, features, target, folder) -> list:
        plots = []
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        cat_features     = [f for f in features if df[f].dtype == object]
        target_is_numeric = target in df.columns and pd.api.types.is_numeric_dtype(df[target])

        corr_cols = numeric_features + ([target] if target_is_numeric and target not in numeric_features else [])
        if len(corr_cols) >= 2:
            url = self._corr_heatmap(df, corr_cols, folder)
            if url:
                plots.append({"title": "Correlation Heatmap", "url": url})

        url = self._target_distribution(df, target, folder)
        if url:
            plots.append({"title": f"Distribution: {target}", "url": url})

        if numeric_features:
            url = self._feature_vs_target(df, numeric_features[0], target, folder)
            if url:
                plots.append({"title": f"{numeric_features[0]} vs {target}", "url": url})

        if cat_features and target_is_numeric:
            url = self._cat_breakdown(df, cat_features[0], target, folder)
            if url:
                plots.append({"title": f"Avg {target} by {cat_features[0]}", "url": url})
        elif len(numeric_features) >= 2:
            url = self._feature_vs_target(df, numeric_features[1], target, folder)
            if url:
                plots.append({"title": f"{numeric_features[1]} vs {target}", "url": url})

        return plots

    def _corr_heatmap(self, df, cols, folder):
        try:
            subset = df[cols].select_dtypes(include="number").dropna()
            subset = subset.loc[:, ~subset.columns.duplicated()]
            if subset.shape[1] < 2:
                return None
            corr = subset.corr()
            fig, ax = plt.subplots(
                figsize=(max(6.5, len(corr) * 0.75), max(5, len(corr) * 0.65)), facecolor=BG
            )
            ax.set_facecolor(PANEL)
            cmap = sns.blend_palette([ACCENT2, PANEL, ACCENT], as_cmap=True)
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 8, "color": "#f8fafc"},
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1
            )
            ax.set_title("Correlation Matrix", fontsize=12, pad=12, color="#f1f5f9")
            ax.tick_params(colors=FG)
            return _save(fig, folder, "heatmap")
        except Exception:
            return None

    def _target_distribution(self, df, target, folder):
        try:
            fig, ax = _new_fig()
            col = df[target].dropna()
            if pd.api.types.is_numeric_dtype(col):
                ax.hist(col, bins=30, color=ACCENT, edgecolor=BG, alpha=0.9)
                ax.set_xlabel(target)
                ax.set_ylabel("Count")
                ax.set_title(f"Distribution of {target}")
            else:
                vc = col.value_counts().head(15)
                bars = ax.barh(vc.index.astype(str), vc.values, color=ACCENT2, alpha=0.9)
                ax.set_xlabel("Count")
                ax.set_title(f"Distribution of {target}")
                ax.invert_yaxis()
                for bar in bars:
                    ax.text(bar.get_width() + max(vc.values) * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            str(int(bar.get_width())), va="center", color=FG, fontsize=8)
            return _save(fig, folder, "dist")
        except Exception:
            return None

    def _feature_vs_target(self, df, feature, target, folder):
        try:
            sample = df[[feature, target]].dropna()
            sample = sample.sample(min(500, len(sample))) if len(sample) > 0 else sample
            fig, ax = _new_fig()
            if pd.api.types.is_numeric_dtype(df[target]) and len(sample) > 1:
                ax.scatter(sample[feature], sample[target], alpha=0.5, color=ACCENT2, edgecolors="none", s=22)
                try:
                    z = np.polyfit(sample[feature], sample[target], 1)
                    p = np.poly1d(z)
                    xs = np.linspace(sample[feature].min(), sample[feature].max(), 200)
                    ax.plot(xs, p(xs), color=ACCENT3, lw=1.8, label="trend")
                    ax.legend(labelcolor=FG, facecolor=PANEL, edgecolor=GRID)
                except Exception:
                    pass
            else:
                means = df.groupby(target)[feature].mean().sort_values().head(12)
                means.plot(kind="barh", ax=ax, color=ACCENT2, alpha=0.9)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title(f"{feature} vs {target}")
            return _save(fig, folder, "scatter")
        except Exception:
            return None

    def _cat_breakdown(self, df, cat_col, num_col, folder):
        try:
            top = df[cat_col].value_counts().head(10).index
            sub = df[df[cat_col].isin(top)]
            means = sub.groupby(cat_col)[num_col].mean().sort_values()
            fig, ax = _new_fig((8, 4.6))
            colors = sns.blend_palette([ACCENT2, ACCENT], n_colors=len(means))
            means.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
            ax.set_xlabel(f"Mean {num_col}")
            ax.set_title(f"Average {num_col} by {cat_col}")
            return _save(fig, folder, "catbar")
        except Exception:
            return None

    def plot_confusion_matrix(self, cm_list, classes, folder):
        try:
            cm = np.array(cm_list)
            fig, ax = plt.subplots(figsize=(max(4.5, len(classes) * 0.9), max(4, len(classes) * 0.8)), facecolor=BG)
            ax.set_facecolor(PANEL)
            cmap = sns.blend_palette([PANEL, ACCENT], as_cmap=True)
            labels = [str(c)[:12] for c in classes]
            sns.heatmap(
                cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels[:len(cm[0])], yticklabels=labels[:len(cm)],
                ax=ax, linewidths=0.5, linecolor=BG,
                annot_kws={"color": "#f8fafc", "size": 10}
            )
            ax.set_xlabel("Predicted", color=FG)
            ax.set_ylabel("Actual", color=FG)
            ax.set_title("Confusion Matrix", color="#f1f5f9", fontsize=12)
            ax.tick_params(colors=FG, rotation=0)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            return _save(fig, folder, "confmat")
        except Exception:
            return None

    def generate_shap(self, model, X_test, features, folder):
        try:
            import shap
            explainer = shap.Explainer(model, X_test)
            shap_vals = explainer(X_test[:100])
            fig = plt.figure(figsize=(8, 4.6), facecolor=BG)
            shap.plots.beeswarm(shap_vals, show=False, max_display=12)
            return _save(fig, folder, "shap")
        except Exception:
            return None
