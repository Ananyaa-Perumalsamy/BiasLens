"""
pages/dashboard.py — Visual Analytics Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

DARK = dict(
    paper_bgcolor="#0a0d14", plot_bgcolor="#111520",
    font=dict(color="#e8ecf4", family="DM Sans"),
    xaxis=dict(gridcolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", zeroline=False),
)


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>📈 Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Visual analytics — group distributions, fairness gauges, outcome rates</div>", unsafe_allow_html=True)

    if "bias_reports" not in st.session_state:
        st.info("Run **Bias Report** first to unlock dashboards.")
        return

    bias_reports   = st.session_state["bias_reports"]
    df_clean       = st.session_state["df_clean"]
    sensitive_cols = st.session_state["sensitive_cols"]
    outcome_col    = st.session_state["outcome_col"]
    summary        = st.session_state["dataset_summary"]

    sc = st.selectbox("View dashboard for attribute", sensitive_cols)
    report = bias_reports[sc]

    # ── Group distribution ────────────────────────────────────
    st.markdown("### Group Distribution in Dataset")
    dist = summary["group_distributions"].get(sc, {})
    fig_dist = go.Figure(go.Bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        marker=dict(color=list(dist.values()), colorscale="Blues", showscale=False),
        text=[f"{v:,}" for v in dist.values()],
        textposition="outside",
    ))
    fig_dist.update_layout(title=f"Sample count per {sc} group",
                            xaxis_title=sc, yaxis_title="Count", **DARK)
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Positive outcome rate per group ───────────────────────
    st.markdown("### Positive Outcome Rate per Group")
    grp_acc = report["group_accuracy"]
    colors  = ["#4f8ef7" if v >= 0.5 else "#f7614f" for v in grp_acc.values()]
    fig_acc = go.Figure(go.Bar(
        x=list(grp_acc.keys()),
        y=[v * 100 for v in grp_acc.values()],
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in grp_acc.values()],
        textposition="outside",
    ))
    fig_acc.add_hline(y=np.mean(list(grp_acc.values())) * 100,
                      line_dash="dash", line_color="#4ff7b8",
                      annotation_text="Overall avg")
    fig_acc.update_layout(title=f"Positive '{outcome_col}' rate by {sc}",
                           xaxis_title=sc, yaxis_title="Rate (%)",
                           yaxis=dict(range=[0, 110], gridcolor="#1e2535", zeroline=False),
                           **{k: v for k, v in DARK.items() if k != "yaxis"})
    st.plotly_chart(fig_acc, use_container_width=True)

    # ── Fairness gauges ───────────────────────────────────────
    st.markdown("### Fairness Metric Gauges")

    def gauge(value, title, ideal_low=True, vmax=1.0, threshold=0.1):
        passed = value <= threshold if ideal_low else value >= threshold
        color  = "#4ff7b8" if passed else "#f7614f"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(value, 4),
            title={"text": title, "font": {"color": "#e8ecf4", "size": 12}},
            gauge={
                "axis":   {"range": [0, vmax], "tickcolor": "#6b7592"},
                "bar":    {"color": color},
                "bgcolor": "#111520",
                "bordercolor": "#1e2535",
                "steps": [
                    {"range": [0, threshold], "color": "#1a2a1a" if not ideal_low else "#1a1a2a"},
                    {"range": [threshold, vmax], "color": "#2a1a1a" if ideal_low else "#1a2a1a"},
                ],
            },
            number={"font": {"color": color, "family": "Space Mono"}},
        ))
        fig.update_layout(paper_bgcolor="#0a0d14", font=dict(color="#e8ecf4"),
                          height=220, margin=dict(l=20, r=20, t=50, b=10))
        return fig

    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(gauge(report["bias_gap"],               "Bias Gap",          ideal_low=True,  vmax=0.5, threshold=0.05), use_container_width=True)
    g2.plotly_chart(gauge(report["demographic_parity_diff"],"Dem. Parity Diff",  ideal_low=True,  vmax=1.0, threshold=0.10), use_container_width=True)
    g3.plotly_chart(gauge(report["equalized_odds"]["equalized_odds_diff"], "Eq. Odds Diff", ideal_low=True, vmax=1.0, threshold=0.10), use_container_width=True)
    g4.plotly_chart(gauge(report["disparate_impact_ratio"], "Disparate Impact",  ideal_low=False, vmax=1.0, threshold=0.80), use_container_width=True)

    # ── TPR/FPR comparison ────────────────────────────────────
    st.markdown("### TPR vs FPR per Group")
    eod    = report["equalized_odds"]
    groups = list(eod["tprs"].keys())
    tprs   = [eod["tprs"][g] * 100 for g in groups]
    fprs   = [eod["fprs"][g] * 100 for g in groups]

    fig_rates = go.Figure()
    fig_rates.add_trace(go.Bar(name="TPR (True Positive Rate)", x=groups, y=tprs, marker_color="#4f8ef7"))
    fig_rates.add_trace(go.Bar(name="FPR (False Positive Rate)", x=groups, y=fprs, marker_color="#f7614f"))
    fig_rates.update_layout(barmode="group", title=f"TPR and FPR by {sc} group",
                             xaxis_title=sc, yaxis_title="%", **DARK)
    st.plotly_chart(fig_rates, use_container_width=True)

    # ── Dataset imbalance ─────────────────────────────────────
    st.markdown("### Dataset Imbalance")
    imb = report["dataset_imbalance"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi² statistic",   f"{imb['chi2']:.2f}")
    col2.metric("p-value",          f"{imb['p_value']:.4f}",
                delta="Imbalanced ⚠️" if imb["is_imbalanced"] else "Balanced ✅",
                delta_color="inverse")
    col3.metric("Imbalance ratio",  f"{imb['imbalance_ratio']:.1f}x")

    # ── Full metrics table ─────────────────────────────────────
    with st.expander("📋 Full Metrics Table"):
        rows = [
            ["Bias Gap",               f"{report['bias_gap']:.4f}",                                              "< 0.05"],
            ["Demographic Parity Diff",f"{report['demographic_parity_diff']:.4f}",                               "= 0"],
            ["Equalized Odds Diff",    f"{report['equalized_odds']['equalized_odds_diff']:.4f}",                 "= 0"],
            ["Disparate Impact Ratio", f"{report['disparate_impact_ratio']:.4f}",                                "≥ 0.8"],
            ["Predictive Parity Gap",  f"{report['predictive_parity']['predictive_parity_gap']:.4f}",            "= 0"],
            ["Imbalance Ratio",        f"{report['dataset_imbalance']['imbalance_ratio']:.2f}x",                 "= 1.0"],
            ["Chi² p-value",           f"{report['dataset_imbalance']['p_value']:.4f}",                          "> 0.05"],
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Ideal"]),
                     use_container_width=True, hide_index=True)
