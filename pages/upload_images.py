"""
pages/upload_images.py — Upload & Configure Image Datasets
Upload image datasets with metadata (CSV or filename-based),
auto-detect sensitive attributes, and prepare for bias analysis.
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import shutil
import zipfile
from pathlib import Path

from core.dataset import (
    load_image_metadata_from_csv,
    load_image_metadata_from_filenames,
    scan_columns,
    build_group_indices,
    dataset_summary,
    encode_outcome,
)


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>🖼️  Upload & Configure Images</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Upload image datasets with metadata — BiasLens prepares them for fairness auditing</div>", unsafe_allow_html=True)

    # ── Step 1: Upload images ─────────────────────────────────
    st.markdown("### Step 1: Upload Images")
    st.markdown("<div style='color:#6b7592;font-size:.85rem;margin-bottom:1rem'>Upload a ZIP file or use an existing folder</div>", unsafe_allow_html=True)

    upload_method = st.radio(
        "Upload method",
        ["Upload ZIP file", "Use existing folder"],
        horizontal=True
    )

    image_dir = None
    temp_dir = None

    if upload_method == "Upload ZIP file":
        uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])
        if uploaded_zip:
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            image_dir = temp_dir

            # Find subfolder if images are in a subfolder
            for root, dirs, files in os.walk(temp_dir):
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if image_files:
                    image_dir = root
                    break

            st.success(f"✅ Extracted ZIP — found {len(image_files)} image files")

    else:
        folder_path = st.text_input(
            "Folder path",
            value="",
            help="Absolute path to folder containing images"
        )
        if folder_path and os.path.isdir(folder_path):
            image_dir = folder_path
            image_count = len([f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            st.success(f"✅ Using folder — found {image_count} image files")

    if image_dir is None:
        st.info("Upload a ZIP or specify a folder path to continue.")
        return

    st.session_state["image_dir"] = image_dir
    st.session_state["temp_dir"] = temp_dir

    # ── Step 2: Choose metadata source ────────────────────────
    st.markdown("---")
    st.markdown("### Step 2: Metadata Source")
    st.markdown("<div style='color:#6b7592;font-size:.85rem;margin-bottom:1rem'>How should we get image attributes?</div>", unsafe_allow_html=True)

    metadata_source = st.radio(
        "Metadata source",
        ["Upload CSV file", "Parse from filenames (UTK format)"],
        horizontal=False
    )

    df = None
    image_paths = None

    if metadata_source == "Upload CSV file":
        st.markdown("#### Upload CSV with metadata")
        st.markdown("<div style='color:#6b7592;font-size:.85rem'>CSV must have 'file' column with image filenames</div>", unsafe_allow_html=True)

        csv_file = st.file_uploader("Upload metadata CSV", type=["csv"], key="metadata_csv")
        if csv_file:
            try:
                df, image_paths = load_image_metadata_from_csv(csv_file, image_dir)
                st.success(f"✅ Loaded metadata for {len(df)} images")
                st.session_state["metadata_source"] = "csv"
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
                return

    else:
        st.markdown("#### Parse from UTK filename format")
        st.markdown("<div style='color:#6b7592;font-size:.85rem'>Expected format: age_gender_race_datetime.jpg</div>", unsafe_allow_html=True)

        if st.button("🔍 Parse filenames"):
            try:
                df, image_paths = load_image_metadata_from_filenames(image_dir)
                st.success(f"✅ Parsed {len(df)} images")
                st.session_state["metadata_source"] = "filenames"
            except Exception as e:
                st.error(f"Failed to parse filenames: {e}")
                return

    if df is None or image_paths is None:
        if "df" in st.session_state and st.session_state.get("dataset_type") == "images":
            df = st.session_state["df"]
            image_paths = st.session_state.get("image_paths", [])
        else:
            st.info("Load metadata to continue.")
            return

    st.session_state["df"] = df
    st.session_state["image_paths"] = image_paths
    st.session_state["dataset_type"] = "images"

    # ── Data preview ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Metadata Preview")
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Images", len(df))
    col2.metric("Attributes", len(df.columns))
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    # ── Auto-scan columns ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Column Configuration")
    st.markdown("<div style='color:#6b7592;font-size:.85rem;margin-bottom:1rem'>BiasLens has auto-detected column roles. Review and adjust as needed.</div>", unsafe_allow_html=True)

    scan = scan_columns(df)

    # Build default selections
    default_sensitive = [
        col for col, info in scan.items()
        if info["role"] == "sensitive" and col != "file"
    ]
    default_outcome = next(
        (col for col, info in scan.items() if info["role"] == "outcome"),
        None
    )

    # Show scan results as a table
    scan_rows = []
    for col, info in scan.items():
        if col == "file":
            continue
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
            options=[c for c in df.columns if c != "file"],
            default=[c for c in default_sensitive if c in df.columns],
            help="Columns representing protected attributes: gender, race, age, etc."
        )

    with col_b:
        outcome_options = [c for c in df.columns if c != "file"]
        outcome_col = st.selectbox(
            "🎯 Outcome / decision column",
            options=outcome_options,
            index=outcome_options.index(default_outcome) if default_outcome in outcome_options else 0,
            help="The column representing the decision or prediction to audit for bias."
        )

    # ── Train/val split ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Train/Validation Split")

    val_split = st.slider(
        "Validation set ratio",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Images will be split with stratification by sensitive groups to maintain fairness"
    )

    # ── Positive outcome class ────────────────────────────────
    st.markdown("---")
    st.markdown("### Which outcome value is the **positive** result?")
    st.markdown("<div style='color:#6b7592;font-size:.82rem;margin-bottom:.5rem'>e.g., '1', 'approved', 'hired' — used for fairness metrics</div>", unsafe_allow_html=True)

    unique_outcomes = df[outcome_col].dropna().unique().tolist()
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

    if st.button("✅  Confirm & Prepare Dataset", type="primary"):
        try:
            # Encode outcome
            y_encoded, label_map = encode_outcome(df, outcome_col)
            pos_class_idx = label_map.get(str(positive_class), 1)

            # Build group indices
            group_indices_all = {}
            for sc in sensitive_cols:
                group_indices_all[sc] = build_group_indices(df, sc)

            # Dataset summary
            summary = dataset_summary(df, sensitive_cols, outcome_col)

            # Store in session
            st.session_state["image_dir"] = image_dir
            st.session_state["image_paths"] = image_paths
            st.session_state["df_clean"] = df
            st.session_state["y_encoded"] = y_encoded
            st.session_state["label_map"] = label_map
            st.session_state["positive_class_idx"] = pos_class_idx
            st.session_state["sensitive_cols"] = sensitive_cols
            st.session_state["outcome_col"] = outcome_col
            st.session_state["group_indices_all"] = group_indices_all
            st.session_state["dataset_summary"] = summary
            st.session_state["val_split"] = val_split
            st.session_state["upload_confirmed"] = True
            st.session_state["dataset_type"] = "images"

            st.success("✅ Dataset prepared! Go to **Train Model** in the sidebar to begin training.")
            st.balloons()

        except Exception as e:
            st.error(f"Error preparing dataset: {e}")
