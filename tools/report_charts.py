"""Deterministic chart generation for the HTML report (Phase 6).

Pure matplotlib/sklearn - no LLM involved, and every function here takes
only numbers/labels already computed by the deterministic pipeline
(tools/evaluation.py's ModelComparison, tools/model_runner.py's chart_data).
Every chart is rendered to a base64-encoded PNG data URI so
reports/template.html can embed it directly (<img src="...">) without any
extra files to serve.

Phase 9: uses matplotlib's object-oriented Figure API directly
(`Figure`/`FigureCanvasAgg`) instead of `pyplot.subplots()`/`pyplot.close()`.
pyplot keeps a global current-figure registry that is not thread-safe -
Phase 9's bounded worker pool runs multiple pipeline runs (each building its
own charts) concurrently in-process, so two runs calling `plt.subplots()` at
the same time could corrupt each other's figure state. Building Figures
directly never touches that global registry, so this is safe under
concurrent report generation.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless - this runs inside the pipeline, never a GUI
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _new_figure(figsize: tuple[float, float]):
    """Figure/Axes via the OO API - never registered with pyplot's global
    state, so nothing needs to be (or can safely be) `plt.close()`-d."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    return fig, ax


def _fig_to_data_uri(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def model_comparison_chart(model_comparison: dict) -> Optional[str]:
    """Bar chart of every successfully-scored candidate on the primary
    metric, winner highlighted. Values come straight from
    model_comparison["results"][*]["metrics"] - nothing computed here."""
    primary_metric = model_comparison.get("primary_metric")
    if not primary_metric:
        return None
    scored = [
        r
        for r in model_comparison.get("results", [])
        if r.get("status") == "success" and r.get("metrics", {}).get(primary_metric) is not None
    ]
    if not scored:
        return None

    winner = model_comparison.get("winner")
    names = [r["model_name"] for r in scored]
    values = [r["metrics"][primary_metric] for r in scored]
    colors = ["#2a9d3f" if n == winner else "#4c6fd6" for n in names]

    fig, ax = _new_figure((7, 4))
    ax.bar(names, values, color=colors)
    ax.set_ylabel(primary_metric)
    ax.set_title(f"Model comparison ({primary_metric})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def historical_vs_predicted_chart(actual: list, predicted: list) -> Optional[str]:
    """Forecasting: the winner's holdout-period actual vs. predicted values,
    in chronological order (chart_data is already time-sorted)."""
    if not actual or not predicted:
        return None
    fig, ax = _new_figure((7, 4))
    x = range(len(actual))
    ax.plot(x, actual, label="Actual", marker="o", markersize=3)
    ax.plot(x, predicted, label="Predicted", marker="x", markersize=3)
    ax.set_title("Historical vs Predicted")
    ax.set_xlabel("Time step (holdout period)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def prediction_vs_actual_chart(actual: list, predicted: list) -> Optional[str]:
    """Regression: scatter of predicted vs. actual with a perfect-prediction
    reference line."""
    if not actual or not predicted:
        return None
    fig, ax = _new_figure((5, 5))
    ax.scatter(actual, predicted, alpha=0.6, s=18)
    lo, hi = min(min(actual), min(predicted)), max(max(actual), max(predicted))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs Actual")
    ax.legend()
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def residual_chart(actual: list, predicted: list) -> Optional[str]:
    """Regression: residual (predicted - actual) per holdout sample."""
    if not actual or not predicted:
        return None
    residuals = [p - a for a, p in zip(actual, predicted)]
    fig, ax = _new_figure((7, 4))
    ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=18)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Holdout sample index")
    ax.set_ylabel("Residual (predicted - actual)")
    ax.set_title("Residual Analysis")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def confusion_matrix_chart(actual: list, predicted: list) -> Optional[str]:
    """Classification: confusion matrix over the winner's holdout predictions."""
    if not actual or not predicted:
        return None
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(actual) | set(predicted))
    cm = confusion_matrix(actual, predicted, labels=labels)

    fig, ax = _new_figure((5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def feature_importance_chart(feature_importance: dict, title: str = "Feature Importance (top 10)") -> Optional[str]:
    """Classification/regression: top-10 importances for one method
    (native/permutation/SHAP - tools/explainability.py), already computed
    and ranked deterministically."""
    if not feature_importance:
        return None
    items = sorted(feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]

    fig, ax = _new_figure((7, 4))
    ax.barh(names, values, color="#4c6fd6")
    ax.set_title(title)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _entries_to_dict(entries: Optional[list]) -> dict:
    """Adapts tools/explainability.py's ExplainabilityResult importance
    lists ([{"feature": ..., "importance": ...}, ...]) into the plain dict
    feature_importance_chart() expects."""
    return {e["feature"]: e["importance"] for e in (entries or []) if isinstance(e, dict) and "feature" in e}


def build_report_charts(problem_type: Optional[str], model_comparison: Optional[dict], chart_data: Optional[dict]) -> dict:
    """Orchestrates which charts to build for the report, based on
    problem_type - "where applicable" per the winning model's own data.
    Never raises: any individual chart failing degrades to that key simply
    being absent, so a plotting issue never breaks report generation.
    """
    charts: dict[str, str] = {}
    if not model_comparison:
        return charts

    try:
        comparison_chart = model_comparison_chart(model_comparison)
        if comparison_chart:
            charts["model_comparison"] = comparison_chart
    except Exception:  # noqa: BLE001 - a chart failing must not break the report
        pass

    winner = model_comparison.get("winner")
    if not winner or not chart_data:
        return charts

    sample = chart_data.get(winner)
    winner_result = next((r for r in model_comparison.get("results", []) if r.get("model_name") == winner), None)

    try:
        if sample:
            actual, predicted = sample.get("actual"), sample.get("predicted")
            if problem_type == "forecasting":
                chart = historical_vs_predicted_chart(actual, predicted)
                if chart:
                    charts["historical_vs_predicted"] = chart
            elif problem_type == "classification":
                chart = confusion_matrix_chart(actual, predicted)
                if chart:
                    charts["confusion_matrix"] = chart
            elif problem_type == "regression":
                pva_chart = prediction_vs_actual_chart(actual, predicted)
                if pva_chart:
                    charts["prediction_vs_actual"] = pva_chart
                residuals = residual_chart(actual, predicted)
                if residuals:
                    charts["residuals"] = residuals
    except Exception:  # noqa: BLE001
        pass

    try:
        feature_importance = winner_result.get("feature_importance") if winner_result else None
        if feature_importance:
            fi_chart = feature_importance_chart(feature_importance)
            if fi_chart:
                charts["feature_importance"] = fi_chart
    except Exception:  # noqa: BLE001
        pass

    try:
        explainability = winner_result.get("explainability") if winner_result else None
        if explainability:
            permutation = _entries_to_dict(explainability.get("permutation_importance"))
            if permutation:
                chart = feature_importance_chart(permutation, title="Permutation Importance (top 10)")
                if chart:
                    charts["permutation_importance"] = chart
            shap_scores = _entries_to_dict(explainability.get("shap_importance"))
            if shap_scores:
                chart = feature_importance_chart(shap_scores, title="SHAP Importance (top 10)")
                if chart:
                    charts["shap_importance"] = chart
    except Exception:  # noqa: BLE001
        pass

    return charts
