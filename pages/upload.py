"""
pages/upload.py — Upload & Configure
The new entry point: upload any CSV, auto-detect columns,
confirm sensitive attributes and outcome column.
"""

import streamlit as st
import pandas as pd
import io
import requests

from core.dataset import scan_columns, dataset_summary, DEMO_DATASETS


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>📂 Upload & Configure</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Upload any CSV dataset — BiasLens auto-detects sensitive attributes and outcome columns</div>", unsafe_allow_html=True)

    # ── Source selector ───────────────────────────────────────
    source = st.radio("Data source", ["Upload my own CSV", "Use a demo dataset"], horizontal=True)

    df = None
    dataset_name = ""

    if source == "Upload my own CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            dataset_name = uploaded.name
            st.success(f"✅ Loaded **{dataset_name}** — {len(df):,} rows × {len(df.columns)} columns")

    else:
        demo_choice = st.selectbox("Choose a demo dataset", list(DEMO_DATASETS.keys()))
        demo_info   = DEMO_DATASETS[demo_choice]

        st.markdown(f"""
        <div class='bl-card' style='font-size:.85rem;color:#6b7592'>
            {demo_info['description']}
        </div>
        """, unsafe_allow_html=True)

        if st.button("⬇️  Load Demo Dataset"):
            with st.spinner("Downloading…"):
                try:
                    resp = requests.get(demo_info["url"], timeout=15)
                    df   = pd.read_csv(io.StringIO(resp.text))
                    dataset_name = demo_choice
                    st.session_state["demo_sensitive_cols"] = demo_info["sensitive_cols"]
                    st.session_state["demo_outcome_col"]    = demo_info["outcome_col"]
                    st.session_state["df"]           = df
                    st.session_state["dataset_name"] = dataset_name
                    st.success(f"✅ Loaded **{demo_choice}** — {len(df):,} rows × {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    return

        # Restore from session if already loaded
        if "df" in st.session_state and st.session_state.get("dataset_name") == demo_choice:
            df           = st.session_state["df"]
            dataset_name = demo_choice

    if df is None:
        if "df" in st.session_state:
            df           = st.session_state["df"]
            dataset_name = st.session_state.get("dataset_name", "dataset")
        else:
            st.info("Upload a CSV or load a demo dataset to continue.")
            return

    # ── Store in session ──────────────────────────────────────
    st.session_state["df"]           = df
    st.session_state["dataset_name"] = dataset_name

    # ── Data preview ──────────────────────────────────────────
    st.markdown("### Data Preview")
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    # ── Auto-scan columns ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Column Configuration")
    st.markdown("<div style='color:#6b7592;font-size:.85rem;margin-bottom:1rem'>BiasLens has auto-detected column roles. Review and adjust as needed.</div>", unsafe_allow_html=True)

    scan = scan_columns(df)

    # Pre-fill from demo defaults if available
    demo_sensitive = st.session_state.get("demo_sensitive_cols", [])
    demo_outcome   = st.session_state.get("demo_outcome_col", "")

    # Build default selections
    default_sensitive = demo_sensitive if demo_sensitive else [
        col for col, info in scan.items() if info["role"] == "sensitive"
    ]
    default_outcome = demo_outcome if demo_outcome else next(
        (col for col, info in scan.items() if info["role"] == "outcome"), df.columns[-1]
    )

    # Show scan results as a table
    scan_rows = []
    for col, info in scan.items():
        confidence_badge = {
            "high":   "🟢 high",
            "medium": "🟡 medium",
            "low":    "🔴 low",
        }.get(info["confidence"], "")
        scan_rows.append({
            "Column":      col,
            "Auto Role":   info["role"],
            "Confidence":  confidence_badge,
            "Type":        info["dtype"],
            "Unique vals": info["nunique"],
            "Samples":     str(info["sample_vals"])[:60],
        })
    st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Confirm your column mapping")

    col_a, col_b = st.columns(2)

    with col_a:
        sensitive_cols = st.multiselect(
            "🔵 Sensitive / demographic columns",
            options=df.columns.tolist(),
            default=[c for c in default_sensitive if c in df.columns],
            help="Columns representing protected attributes: gender, race, age, etc."
        )

    with col_b:
        outcome_col = st.selectbox(
            "🎯 Outcome / decision column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_outcome) if default_outcome in df.columns else len(df.columns) - 1,
            help="The column representing the decision or prediction to audit for bias."
        )

    # ── Outcome value mapping ─────────────────────────────────
    if outcome_col:
        unique_outcomes = df[outcome_col].dropna().unique().tolist()
        st.markdown("#### Which outcome value is the **positive** result?")
        st.markdown("<div style='color:#6b7592;font-size:.82rem;margin-bottom:.5rem'>e.g. 'hired', 'approved', '1', '>50K'</div>", unsafe_allow_html=True)
        positive_class = st.selectbox(
            "Positive outcome",
            options=[str(v) for v in unique_outcomes],
            label_visibility="collapsed"
        )

    # ── Confirm ───────────────────────────────────────────────
    st.markdown("---")

    if not sensitive_cols:
        st.warning("Select at least one sensitive column to continue.")
        return
    if not outcome_col:
        st.warning("Select an outcome column to continue.")
        return

    if st.button("✅  Confirm & Proceed to Bias Analysis", type="primary"):
        # Encode outcome
        from core.dataset import encode_outcome, build_group_indices, dataset_summary

        df_clean = df.dropna(subset=[outcome_col] + sensitive_cols).copy()
        df_clean[outcome_col] = df_clean[outcome_col].astype(str)

        y_encoded, label_map = encode_outcome(df_clean, outcome_col)
        pos_class_idx = label_map.get(str(positive_class), 1)

        group_indices_all = {}
        for sc in sensitive_cols:
            group_indices_all[sc] = build_group_indices(df_clean, sc)

        summary = dataset_summary(df_clean, sensitive_cols, outcome_col)

        st.session_state["df_clean"]          = df_clean
        st.session_state["y_encoded"]         = y_encoded
        st.session_state["label_map"]         = label_map
        st.session_state["positive_class_idx"] = pos_class_idx
        st.session_state["sensitive_cols"]    = sensitive_cols
        st.session_state["outcome_col"]       = outcome_col
        st.session_state["group_indices_all"] = group_indices_all
        st.session_state["dataset_summary"]   = summary
        st.session_state["upload_confirmed"]  = True

        # Clear any stale bias report
        st.session_state.pop("bias_report", None)

        st.success("✅ Configuration saved! Go to **Bias Report** in the sidebar.")
        st.balloons()
