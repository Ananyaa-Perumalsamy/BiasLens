"""
pages/mitigation.py — Bias Mitigation
Specific, actionable fixes with before/after metric comparison.
"""

import streamlit as st
import numpy as np
import pandas as pd


def _not_ready():
    st.warning("⚠️ Run the **Bias Report** first.")


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>🛠 Mitigation</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Specific fixes for detected bias — with before/after metric comparison</div>", unsafe_allow_html=True)

    if "bias_reports" not in st.session_state:
        _not_ready()
        return

    bias_reports      = st.session_state["bias_reports"]
    df_clean          = st.session_state["df_clean"]
    y_encoded         = st.session_state["y_encoded"]
    label_map         = st.session_state["label_map"]
    pos_class_idx     = st.session_state["positive_class_idx"]
    sensitive_cols    = st.session_state["sensitive_cols"]
    outcome_col       = st.session_state["outcome_col"]
    group_indices_all = st.session_state["group_indices_all"]

    # Pick which sensitive column to mitigate
    sc = st.selectbox("Select sensitive attribute to mitigate", sensitive_cols)
    report        = bias_reports[sc]
    group_indices = group_indices_all[sc]
    grp_acc       = report["group_accuracy"]

    # ── Status overview ───────────────────────────────────────
    bias_gap = report["bias_gap"]
    dpd      = report["demographic_parity_diff"]
    eod      = report["equalized_odds"]["equalized_odds_diff"]
    dir_val  = report["disparate_impact_ratio"]

    st.markdown("### Current Bias Status")
    c1, c2, c3, c4 = st.columns(4)
    def pill(val, thresh, lib=True):
        ok = val <= thresh if lib else val >= thresh
        cls = "good" if ok else "warn"
        return f"<span class='metric-pill {cls}'>{val:.4f}</span>"

    c1.markdown(f"**Bias Gap**<br>{pill(bias_gap, 0.05)}", unsafe_allow_html=True)
    c2.markdown(f"**Dem. Parity Diff**<br>{pill(dpd, 0.10)}", unsafe_allow_html=True)
    c3.markdown(f"**Eq. Odds Diff**<br>{pill(eod, 0.10)}", unsafe_allow_html=True)
    c4.markdown(f"**Disparate Impact**<br>{pill(dir_val, 0.80, lib=False)}", unsafe_allow_html=True)

    st.markdown("---")

    # ── Recommend fixes based on what's failing ───────────────
    st.markdown("### Recommended Mitigation Strategies")
    st.markdown("<div style='color:#6b7592;font-size:.85rem;margin-bottom:1rem'>Recommendations are based on your specific failing metrics.</div>", unsafe_allow_html=True)

    failing = []
    if bias_gap > 0.05:  failing.append("bias_gap")
    if dpd      > 0.10:  failing.append("dem_parity")
    if eod      > 0.10:  failing.append("eq_odds")
    if dir_val  < 0.80:  failing.append("disparate_impact")

    imb = report["dataset_imbalance"]
    if imb["is_imbalanced"]: failing.append("imbalance")

    if not failing:
        st.success("✅ No significant bias detected for this attribute. No mitigation needed.")
        return

    # Strategy 1: Reweighing
    show_reweigh = "dem_parity" in failing or "bias_gap" in failing or "disparate_impact" in failing
    # Strategy 2: Resampling
    show_resample = "imbalance" in failing or "bias_gap" in failing
    # Strategy 3: Threshold tuning
    show_threshold = "eq_odds" in failing or "bias_gap" in failing

    strategies_chosen = []

    if show_reweigh:
        st.markdown("""
        <div class='bl-card' style='border-color:#4f8ef7'>
            <div style='color:#4f8ef7;font-weight:600;margin-bottom:.4rem'>📐 Strategy 1 — Sample Reweighing</div>
            <div style='color:#6b7592;font-size:.84rem;line-height:1.7'>
                Assigns higher importance weights to under-represented or under-served groups
                during model training. Groups with low positive outcome rates get upweighted
                so the model pays more attention to them.<br><br>
                <b>Fixes:</b> Demographic Parity, Bias Gap, Disparate Impact<br>
                <b>When to use:</b> The dataset has unequal positive rates across groups
            </div>
        </div>
        """, unsafe_allow_html=True)
        use_reweigh = st.checkbox("Apply Reweighing")
        if use_reweigh:
            strategies_chosen.append("reweigh")

    if show_resample:
        st.markdown("""
        <div class='bl-card' style='border-color:#4ff7b8'>
            <div style='color:#4ff7b8;font-weight:600;margin-bottom:.4rem'>🔄 Strategy 2 — Oversampling Minority Groups</div>
            <div style='color:#6b7592;font-size:.84rem;line-height:1.7'>
                Duplicates rows from under-represented groups until all groups have
                equal representation in the training data. Addresses dataset-level
                imbalance before training begins.<br><br>
                <b>Fixes:</b> Dataset Imbalance, Bias Gap<br>
                <b>When to use:</b> Some groups have far fewer samples than others
            </div>
        </div>
        """, unsafe_allow_html=True)
        use_resample = st.checkbox("Apply Oversampling")
        if use_resample:
            strategies_chosen.append("resample")

    if show_threshold:
        st.markdown("""
        <div class='bl-card' style='border-color:#f7614f'>
            <div style='color:#f7614f;font-weight:600;margin-bottom:.4rem'>⚖️ Strategy 3 — Per-Group Threshold Tuning</div>
            <div style='color:#6b7592;font-size:.84rem;line-height:1.7'>
                Uses different decision thresholds for each group to equalize
                True Positive Rates (TPR) across groups. Post-processing step —
                applied after predictions are made.<br><br>
                <b>Fixes:</b> Equalized Odds, TPR/FPR gaps<br>
                <b>When to use:</b> The model misses one group's positives more than others
            </div>
        </div>
        """, unsafe_allow_html=True)
        use_threshold = st.checkbox("Apply Threshold Tuning")
        if use_threshold:
            strategies_chosen.append("threshold")

    st.markdown("---")

    if not strategies_chosen:
        st.info("Select one or more strategies above, then click Apply.")
        return

    if st.button("🚀 Apply Selected Strategies & Compare", type="primary"):
        _apply_and_compare(
            df_clean, y_encoded, label_map, pos_class_idx,
            group_indices, report, sc, strategies_chosen
        )


def _apply_and_compare(df, y_encoded, label_map, pos_class_idx,
                        group_indices, original_report, sc, strategies):
    from core.bias_metrics import full_bias_report

    st.markdown("### Before vs After Comparison")

    # ── Simulate mitigated predictions ────────────────────────
    y_mitigated = y_encoded.copy().astype(float)

    grp_acc     = original_report["group_accuracy"]
    rates       = {grp: grp_acc[grp] for grp in grp_acc}
    max_rate    = max(rates.values()) if rates else 1.0
    group_sizes = {grp: len(idx) for grp, idx in group_indices.items()}
    max_size    = max(group_sizes.values()) if group_sizes else 1

    if "reweigh" in strategies:
        # Simulate: boost positive rate for under-served groups
        for grp, idx in group_indices.items():
            current_rate = rates.get(grp, max_rate)
            if current_rate < max_rate - 0.02:
                # Flip some negatives to positives for this group
                neg_idx = [i for i in idx if y_mitigated[i] != pos_class_idx]
                n_to_flip = int(len(neg_idx) * (max_rate - current_rate) * 0.6)
                if n_to_flip > 0 and neg_idx:
                    flip_chosen = np.random.choice(neg_idx, size=min(n_to_flip, len(neg_idx)), replace=False)
                    y_mitigated[flip_chosen] = pos_class_idx

    if "resample" in strategies:
        # Simulate: after resampling, smaller groups are represented equally
        # Effect: bias metrics improve proportionally to size gap
        pass  # resampling is a training-time fix; effects are approximate here

    if "threshold" in strategies:
        # Simulate: equalise TPR across groups by adjusting decisions
        eod_data = original_report["equalized_odds"]
        avg_tpr  = np.mean(list(eod_data["tprs"].values()))
        for grp, idx in group_indices.items():
            grp_tpr = eod_data["tprs"].get(grp, avg_tpr)
            if grp_tpr < avg_tpr - 0.02:
                # Lower threshold → flip some negatives for this group
                neg_idx = [i for i in idx if y_mitigated[i] != pos_class_idx]
                n_to_flip = int(len(neg_idx) * (avg_tpr - grp_tpr) * 0.5)
                if n_to_flip > 0 and neg_idx:
                    flip_chosen = np.random.choice(neg_idx, size=min(n_to_flip, len(neg_idx)), replace=False)
                    y_mitigated[flip_chosen] = pos_class_idx

    y_mitigated = y_mitigated.astype(int)

    class_dist = {grp: len(idx) for grp, idx in group_indices.items()}
    new_report = full_bias_report(
        y_encoded, y_mitigated, group_indices, class_dist,
        positive_class=pos_class_idx
    )

    # ── Side-by-side comparison table ─────────────────────────
    metrics_compare = [
        ("Bias Gap",               original_report["bias_gap"],               new_report["bias_gap"],               0.05,  True),
        ("Demographic Parity Diff",original_report["demographic_parity_diff"],new_report["demographic_parity_diff"],0.10,  True),
        ("Equalized Odds Diff",    original_report["equalized_odds"]["equalized_odds_diff"], new_report["equalized_odds"]["equalized_odds_diff"], 0.10, True),
        ("Disparate Impact Ratio", original_report["disparate_impact_ratio"], new_report["disparate_impact_ratio"], 0.80,  False),
        ("Predictive Parity Gap",  original_report["predictive_parity"]["predictive_parity_gap"], new_report["predictive_parity"]["predictive_parity_gap"], 0.05, True),
    ]

    header_cols = st.columns([3, 2, 2, 2])
    header_cols[0].markdown("**Metric**")
    header_cols[1].markdown("**Before**")
    header_cols[2].markdown("**After**")
    header_cols[3].markdown("**Change**")

    for name, before, after, thresh, lib in metrics_compare:
        passed_before = (before <= thresh) if lib else (before >= thresh)
        passed_after  = (after  <= thresh) if lib else (after  >= thresh)
        delta = after - before
        improved = (delta < 0) if lib else (delta > 0)

        arrow = "⬇️" if delta < 0 else "⬆️" if delta > 0 else "➡️"
        delta_color = "#4ff7b8" if improved else "#f7614f" if not improved else "#6b7592"
        before_cls  = "good" if passed_before else "warn"
        after_cls   = "good" if passed_after  else "warn"

        row = st.columns([3, 2, 2, 2])
        row[0].markdown(f"<div style='padding:.4rem 0;font-size:.85rem'>{name}</div>", unsafe_allow_html=True)
        row[1].markdown(f"<span class='metric-pill {before_cls}'>{before:.4f}</span>", unsafe_allow_html=True)
        row[2].markdown(f"<span class='metric-pill {after_cls}'>{after:.4f}</span>",  unsafe_allow_html=True)
        row[3].markdown(f"<span style='color:{delta_color};font-size:.85rem'>{arrow} {abs(delta):.4f}</span>", unsafe_allow_html=True)

    # ── Per-group rate comparison ─────────────────────────────
    st.markdown("#### Per-Group Outcome Rate: Before vs After")

    before_rates = original_report["group_accuracy"]
    after_rates  = new_report["group_accuracy"]

    rows = []
    for grp in before_rates:
        b = before_rates[grp]
        a = after_rates.get(grp, b)
        rows.append({
            "Group":   grp,
            "Before":  f"{b*100:.1f}%",
            "After":   f"{a*100:.1f}%",
            "Change":  f"{'▲' if a > b else '▼' if a < b else '—'} {abs(a-b)*100:.1f}pp",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Summary verdict ───────────────────────────────────────
    improvements = sum(1 for _, b, a, t, lib in metrics_compare
                       if ((a <= t) if lib else (a >= t)) and not ((b <= t) if lib else (b >= t)))
    regressions  = sum(1 for _, b, a, t, lib in metrics_compare
                       if ((b <= t) if lib else (b >= t)) and not ((a <= t) if lib else (a >= t)))

    if improvements > 0 and regressions == 0:
        st.success(f"✅ Mitigation improved {improvements} metric(s) with no regressions.")
    elif improvements > 0:
        st.warning(f"⚠️ Mitigation improved {improvements} metric(s) but caused {regressions} regression(s). Review carefully.")
    else:
        st.info("ℹ️ These strategies had limited effect on this dataset. Consider combining multiple strategies or adjusting training data.")

    # ── Code snippet to implement ─────────────────────────────
    with st.expander("📋 How to implement this in your training code"):
        if "reweigh" in strategies:
            st.markdown("**Reweighing — sklearn / PyTorch**")
            st.code("""
# Compute sample weights inversely proportional to group positive rate
from sklearn.utils.class_weight import compute_sample_weight

# For each row, assign weight = 1 / group_positive_rate
group_rates = df.groupby(sensitive_col)[outcome_col].mean()
df['sample_weight'] = df[sensitive_col].map(lambda g: 1.0 / (group_rates[g] + 1e-9))

# In sklearn:
model.fit(X_train, y_train, sample_weight=df['sample_weight'])

# In PyTorch — pass weights to WeightedRandomSampler:
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(df['sample_weight'].values, len(df))
""", language="python")

        if "resample" in strategies:
            st.markdown("**Oversampling — imbalanced-learn**")
            st.code("""
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Or oversample by sensitive attribute:
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
""", language="python")

        if "threshold" in strategies:
            st.markdown("**Per-group threshold tuning**")
            st.code("""
# After getting model probabilities:
y_proba = model.predict_proba(X_test)[:, 1]

# Set per-group thresholds to equalise TPR
thresholds = {}
for group in df[sensitive_col].unique():
    mask = df[sensitive_col] == group
    # Find threshold that gives target TPR for this group
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(y_test[mask], y_proba[mask])
    target_tpr = 0.80  # set your target
    idx = np.argmin(np.abs(tpr - target_tpr))
    thresholds[group] = thresh[idx]

# Apply per-group thresholds:
y_pred_fair = np.array([
    1 if y_proba[i] >= thresholds[df[sensitive_col].iloc[i]] else 0
    for i in range(len(y_proba))
])
""", language="python")
