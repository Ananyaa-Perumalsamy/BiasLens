"""
pages/bias.py — Bias Report
Works directly on uploaded CSV data.
No model training required.
"""

import streamlit as st
import numpy as np
import pandas as pd


def _not_ready():
    st.warning("⚠️ No dataset configured. Go to **Upload & Configure** first.")


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>📊 Bias Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Comprehensive fairness audit across all sensitive attributes</div>", unsafe_allow_html=True)

    if not st.session_state.get("upload_confirmed"):
        _not_ready()
        return

    df_clean          = st.session_state["df_clean"]
    y_encoded         = st.session_state["y_encoded"]
    label_map         = st.session_state["label_map"]
    pos_class_idx     = st.session_state["positive_class_idx"]
    sensitive_cols    = st.session_state["sensitive_cols"]
    outcome_col       = st.session_state["outcome_col"]
    group_indices_all = st.session_state["group_indices_all"]
    summary           = st.session_state["dataset_summary"]
    dataset_name      = st.session_state.get("dataset_name", "dataset")

    # ── Dataset summary ───────────────────────────────────────
    st.markdown("### Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",             f"{summary['rows']:,}")
    c2.metric("Sensitive cols",   len(sensitive_cols))
    c3.metric("Outcome column",   outcome_col)
    c4.metric("Missing values",   summary["missing"])

    # Outcome distribution
    st.markdown("#### Outcome Distribution")
    outcome_dist = summary["outcome_distribution"]
    total = sum(outcome_dist.values())
    cols = st.columns(len(outcome_dist))
    for col, (label, count) in zip(cols, outcome_dist.items()):
        col.metric(str(label), f"{count:,}", f"{count/total*100:.1f}%")

    st.markdown("---")

    # ── Per sensitive column analysis ─────────────────────────
    from core.bias_metrics import full_bias_report

    all_reports = {}

    for sc in sensitive_cols:
        group_indices = group_indices_all[sc]
        class_dist    = {grp: len(idx) for grp, idx in group_indices.items()}

        # For tabular data: y_true = y_encoded (actual labels),
        # y_pred = same (since we're auditing the decisions in the data itself,
        # not a separate model — the data IS the decision)
        # This analyses the distribution of decisions across groups directly.
        y_true = y_encoded
        y_pred = y_encoded   # auditing raw decisions, not a model's predictions

        report = full_bias_report(
            y_true, y_pred, group_indices, class_dist,
            positive_class=pos_class_idx
        )
        all_reports[sc] = report

    st.session_state["bias_reports"] = all_reports
    st.session_state["bias_report"]  = all_reports[sensitive_cols[0]]  # for mitigation page compat

    # ── Display per sensitive column ──────────────────────────
    for sc in sensitive_cols:
        report = all_reports[sc]

        st.markdown(f"## Sensitive attribute: `{sc}`")

        group_dist = summary["group_distributions"].get(sc, {})
        total_rows = sum(group_dist.values()) or 1

        # Group representation
        st.markdown("#### Group Representation in Dataset")
        dist_cols = st.columns(min(len(group_dist), 6))
        for col, (grp, cnt) in zip(dist_cols, group_dist.items()):
            pct = cnt / total_rows * 100
            pill_cls = "warn" if pct < 10 else "good" if pct > 30 else ""
            col.markdown(f"""
            <div class='bl-card' style='text-align:center;padding:1rem'>
                <div style='font-weight:600;font-size:.9rem'>{grp}</div>
                <div class='metric-pill {pill_cls}'>{pct:.1f}%</div>
                <div style='color:#6b7592;font-size:.75rem'>{cnt:,} rows</div>
            </div>
            """, unsafe_allow_html=True)

        # Positive outcome rate per group
        st.markdown("#### Positive Outcome Rate per Group")
        grp_acc = report["group_accuracy"]
        rate_cols = st.columns(min(len(grp_acc), 6))
        rates_list = list(grp_acc.values())
        min_rate = min(rates_list) if rates_list else 0
        max_rate = max(rates_list) if rates_list else 1

        for col, (grp, rate) in zip(rate_cols, grp_acc.items()):
            is_min = abs(rate - min_rate) < 1e-6 and len(rates_list) > 1
            is_max = abs(rate - max_rate) < 1e-6 and len(rates_list) > 1
            pill_cls = "warn" if is_min else "good" if is_max else ""
            col.markdown(f"""
            <div class='bl-card' style='text-align:center;padding:1rem'>
                <div style='font-weight:600;font-size:.9rem'>{grp}</div>
                <div class='metric-pill {pill_cls}'>{rate*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Fairness metrics
        st.markdown("#### Fairness Metrics")

        metrics = [
            ("Bias Gap",               report["bias_gap"],                                   0.05,  True,  "Max rate − min rate across groups. Ideal: < 0.05"),
            ("Demographic Parity Diff",report["demographic_parity_diff"],                    0.10,  True,  "Gap in positive prediction rates. Ideal: 0"),
            ("Equalized Odds Diff",    report["equalized_odds"]["equalized_odds_diff"],      0.10,  True,  "TPR gap + FPR gap across groups. Ideal: 0"),
            ("Disparate Impact Ratio", report["disparate_impact_ratio"],                     0.80,  False, "Min rate / max rate. ≥ 0.8 required (legal standard)"),
            ("Predictive Parity Gap",  report["predictive_parity"]["predictive_parity_gap"],0.05,  True,  "Precision gap across groups. Ideal: 0"),
        ]

        m_cols = st.columns(len(metrics))
        for col, (name, val, threshold, lower_is_better, note) in zip(m_cols, metrics):
            if lower_is_better:
                passed = val <= threshold
            else:
                passed = val >= threshold

            status     = "good" if passed else "warn"
            verdict    = "✅ Fair" if passed else "❌ Biased"
            verdict_color = "#4ff7b8" if passed else "#f7614f"

            col.markdown(f"""
            <div class='bl-card' style='text-align:center'>
                <div style='color:#6b7592;font-size:.72rem;margin-bottom:.4rem'>{name}</div>
                <div class='metric-pill {status}' style='font-size:1rem'>{val:.4f}</div>
                <div style='color:{verdict_color};font-size:.78rem;margin-top:.4rem;font-weight:600'>{verdict}</div>
                <div style='color:#6b7592;font-size:.68rem;margin-top:.2rem'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

        # Chi-square imbalance
        imb = report["dataset_imbalance"]
        imb_status = "warn" if imb["is_imbalanced"] else "good"
        imb_verdict = "⚠️ Dataset is imbalanced" if imb["is_imbalanced"] else "✅ Dataset is balanced"

        st.markdown(f"""
        <div class='bl-card' style='display:flex;gap:2rem;align-items:center'>
            <div>
                <div style='color:#6b7592;font-size:.8rem'>Dataset Imbalance (Chi²)</div>
                <div class='metric-pill {imb_status}'>χ² = {imb['chi2']:.2f} &nbsp;|&nbsp; p = {imb['p_value']:.4f} &nbsp;|&nbsp; ratio = {imb['imbalance_ratio']:.1f}x</div>
            </div>
            <div style='font-size:.85rem;font-weight:600'>{imb_verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        # Plain-English verdict
        st.markdown("#### Plain-English Summary")
        failed = [name for name, val, threshold, lib, _ in metrics
                  if (val > threshold if lib else val < threshold)]
        passed_count = len(metrics) - len(failed)

        if not failed:
            st.success(f"✅ **{sc}** passes all {len(metrics)} fairness checks. No significant bias detected.")
        else:
            st.error(f"❌ **{sc}** fails {len(failed)} of {len(metrics)} fairness checks: **{', '.join(failed)}**")
            _plain_english_explanations(sc, failed, report, df_clean, outcome_col)

        # Detailed TPR/FPR table
        with st.expander(f"Detailed per-group rates for `{sc}`"):
            eod = report["equalized_odds"]
            rows = []
            for grp in eod["tprs"]:
                rows.append({
                    "Group": grp,
                    "Positive Rate": f"{grp_acc.get(grp, 0)*100:.1f}%",
                    "TPR":   f"{eod['tprs'][grp]*100:.1f}%",
                    "FPR":   f"{eod['fprs'][grp]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")

    st.success("✅ Bias report complete. Go to **Mitigation** to apply fixes.")


def _plain_english_explanations(sc, failed_metrics, report, df, outcome_col):
    grp_acc = report["group_accuracy"]
    sorted_groups = sorted(grp_acc.items(), key=lambda x: x[1])
    worst_group = sorted_groups[0][0]  if sorted_groups else "a group"
    best_group  = sorted_groups[-1][0] if sorted_groups else "another group"

    explanations = {
        "Bias Gap": f"The gap between the highest and lowest outcome rates across `{sc}` groups is too large. "
                    f"Group **{worst_group}** receives the positive outcome only **{grp_acc.get(worst_group,0)*100:.1f}%** of the time, "
                    f"vs **{grp_acc.get(best_group,0)*100:.1f}%** for **{best_group}**.",

        "Demographic Parity Diff": f"The rate of positive outcomes is significantly unequal across `{sc}` groups. "
                                    f"This means the dataset is systematically giving `{outcome_col}` more to some groups than others, "
                                    f"regardless of whether that is justified.",

        "Equalized Odds Diff": f"The model makes different types of errors for different `{sc}` groups — "
                                f"some groups are more likely to be falsely included or falsely excluded. "
                                f"TPR gap: {report['equalized_odds']['tpr_gap']:.3f}, FPR gap: {report['equalized_odds']['fpr_gap']:.3f}.",

        "Predictive Parity Gap": f"When a positive outcome is predicted, it is correct at different rates for different `{sc}` groups. "
                                  f"The tool is less reliable for some groups than others.",
    }

    for metric in failed_metrics:
        if metric in explanations:
            st.markdown(f"""
            <div class='bl-card' style='border-color:#f7614f'>
                <div style='color:#f7614f;font-size:.78rem;font-weight:600;margin-bottom:.3rem'>❌ {metric}</div>
                <div style='color:#6b7592;font-size:.84rem'>{explanations[metric]}</div>
            </div>
            """, unsafe_allow_html=True)
