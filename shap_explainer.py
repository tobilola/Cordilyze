"""
Cordilyze SHAP Explainability Module
=====================================

Replaces hand-coded feature contribution heuristics with real,
model-derived SHAP (SHapley Additive exPlanations) values.

This is a drop-in module. Import it and call the functions from
your app to get actual feature importances from your trained models.

What this gives you:
  - Per-patient explanations: "For THIS patient, blood pressure
    contributed +12 points to their risk score"
  - Global feature importance: "Across all patients, blood pressure
    is the #1 risk driver"
  - Investor-grade explainability (SHAP is the industry standard
    used by banks, hospitals, and FDA-cleared AI tools)

Usage in cordilyze_app.py:
  from shap_explainer import explain_prediction, create_shap_waterfall, create_shap_bar
"""

import numpy as np
import shap
import plotly.graph_objects as go
import streamlit as st

# Feature names matching your model's training order
FEATURE_NAMES = [
    'Age', 'Sex', 'Total Cholesterol', 'HDL Cholesterol',
    'LDL Cholesterol', 'Triglycerides', 'Systolic BP', 'Diastolic BP',
    'Glucose', 'BMI', 'Smoking', 'Physical Activity'
]


@st.cache_resource
def get_shap_explainer(_xgb_model):
    """
    Create and cache a SHAP TreeExplainer for the XGBoost model.

    We use XGBoost because:
      1. It gets the highest ensemble weight (0.40)
      2. TreeExplainer is exact for tree models (not approximate)
      3. It's fast — adds ~20ms per explanation

    The explainer is cached so it's only created once per session.
    """
    explainer = shap.TreeExplainer(_xgb_model)
    return explainer


def explain_prediction(xgb_model, patient_features):
    """
    Compute SHAP values for a single patient.

    Args:
        xgb_model:        Trained XGBoost model
        patient_features:  np.array of shape (1, 12) — same order as predict_risk()

    Returns:
        dict with keys:
          - 'shap_values':    np.array of shape (12,) — contribution of each feature
          - 'base_value':     float — the model's average prediction (baseline)
          - 'contributions':  dict mapping feature name → contribution percentage
          - 'top_drivers':    list of (feature_name, contribution) sorted by |impact|
    """
    explainer = get_shap_explainer(xgb_model)

    # SHAP values for the positive class (CVD risk)
    shap_values = explainer.shap_values(patient_features)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classifier returns [class_0_shap, class_1_shap]
        sv = shap_values[1][0]  # class 1 (CVD positive), first sample
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]  # class 1

    # Convert raw SHAP values to percentage contributions
    # Normalize so they sum to ~100 for easy interpretation
    abs_shap = np.abs(sv)
    total = abs_shap.sum()

    if total > 0:
        contributions = {
            FEATURE_NAMES[i]: round(float(abs_shap[i] / total * 100), 1)
            for i in range(len(FEATURE_NAMES))
        }
    else:
        contributions = {name: 0.0 for name in FEATURE_NAMES}

    # Sort by absolute impact
    top_drivers = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'shap_values': sv,
        'base_value': float(base_value),
        'contributions': contributions,
        'top_drivers': top_drivers,
    }


def create_shap_waterfall(explanation, risk_score):
    """
    Create a Plotly waterfall chart showing how each feature pushes
    the risk score up or down from the baseline.

    This is the "money chart" for investors — it shows exactly WHY
    a patient got their score, feature by feature.
    """
    sv = explanation['shap_values']
    base = explanation['base_value']

    # Sort features by absolute SHAP value (largest impact first)
    indices = np.argsort(np.abs(sv))[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    sorted_values = [float(sv[i]) for i in indices]

    # Top 8 features (keep chart clean)
    sorted_names = sorted_names[:8]
    sorted_values = sorted_values[:8]

    # Build waterfall
    colors = ['#ef4444' if v > 0 else '#10b981' for v in sorted_values]

    fig = go.Figure(go.Waterfall(
        name="Risk Factors",
        orientation="h",
        y=sorted_names[::-1],  # reverse so biggest is on top
        x=sorted_values[::-1],
        connector={"line": {"color": "#e2e8f0", "width": 1}},
        decreasing={"marker": {"color": "#10b981"}},  # green = reduces risk
        increasing={"marker": {"color": "#ef4444"}},   # red = increases risk
        totals={"marker": {"color": "#2563eb"}},
        textposition="outside",
        text=[f'{v:+.3f}' for v in sorted_values[::-1]],
    ))

    fig.update_layout(
        title=dict(
            text="How Each Factor Affects Your Risk",
            font=dict(size=20, color='#1e40af')
        ),
        xaxis_title="Impact on Risk (SHAP value)",
        yaxis_title="",
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        showlegend=False,
        margin=dict(l=140, r=60, t=60, b=50),
    )

    # Add annotation explaining direction
    fig.add_annotation(
        text="← Lowers Risk    |    Raises Risk →",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, color="#64748b"),
    )

    return fig


def create_shap_bar(explanation):
    """
    Create a horizontal bar chart of feature importance percentages.

    Cleaner replacement for the old hand-coded create_feature_importance_chart().
    """
    drivers = explanation['top_drivers']
    features = [d[0] for d in drivers]
    importances = [d[1] for d in drivers]

    # Color by contribution level
    colors = [
        '#ef4444' if imp > 20 else '#f59e0b' if imp > 10 else '#2563eb'
        for imp in importances
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=[f'{imp:.1f}%' for imp in importances],
        textposition='outside',
        hovertemplate='%{y}<br>Contribution: %{x:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text="What's Driving Your Risk? (SHAP Analysis)",
            font=dict(size=20, color='#1e40af')
        ),
        xaxis_title="Contribution to Risk (%)",
        yaxis_title="",
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        xaxis=dict(range=[0, max(importances) * 1.25 if importances else 100]),
        showlegend=False,
        margin=dict(l=140, r=60, t=60, b=50),
        yaxis=dict(autorange="reversed"),  # biggest on top
    )

    return fig


def create_shap_comparison(explanation_before, explanation_after, label_before="Current", label_after="After Changes"):
    """
    Side-by-side SHAP comparison for the What-If Simulator.

    Shows how the SHAP contributions shift when a patient adjusts
    lifestyle factors — makes the simulator dramatically more convincing
    because you can see WHICH factors drove the improvement.
    """
    features = FEATURE_NAMES
    before_vals = [explanation_before['contributions'].get(f, 0) for f in features]
    after_vals = [explanation_after['contributions'].get(f, 0) for f in features]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=label_before,
        x=before_vals,
        y=features,
        orientation='h',
        marker_color='#cbd5e1',
        text=[f'{v:.1f}%' for v in before_vals],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name=label_after,
        x=after_vals,
        y=features,
        orientation='h',
        marker_color='#2563eb',
        text=[f'{v:.1f}%' for v in after_vals],
        textposition='outside',
    ))

    fig.update_layout(
        title="Risk Factor Shift: Before vs After",
        xaxis_title="Contribution to Risk (%)",
        barmode='group',
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        margin=dict(l=140, r=60, t=60, b=50),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
