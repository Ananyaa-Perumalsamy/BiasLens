"""
pages/train.py — Image Model Training (optional)
Works with ANY image dataset + CSV labels.
FairFace is just one example — not required.
"""

import os
import streamlit as st
import torch


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>🧠 Train Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Optional — for image datasets. Train a ResNet-18 classifier, then audit it for bias with Grad-CAM.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='bl-card' style='border-color:#4f8ef7'>
        <div style='color:#4f8ef7;font-weight:600;font-size:.85rem;margin-bottom:.4rem'>ℹ️ When do you need this page?</div>
        <div style='color:#6b7592;font-size:.83rem;line-height:1.7'>
            This page is only needed if you have an <b>image dataset</b> (e.g. face photos).
            For tabular CSV data (hiring, loans, healthcare), go directly to
            <b>Upload & Configure → Bias Report</b> — no training needed.<br><br>
            This page accepts <b>any</b> image dataset — FairFace, UTKFace, CelebA, or your own.
            First upload images via <b>Upload Images</b>, then train here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check if data prepared via upload_images page ──────────
    has_prepared_data = (
        st.session_state.get("dataset_type") == "images"
        and st.session_state.get("upload_confirmed")
    )

    if not has_prepared_data:
        st.info("📂 First, go to **Upload Images** to prepare your dataset.")
        return

    # ── Training config ───────────────────────────────────────
    st.markdown("### Training Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs     = st.slider("Epochs", 1, 30, 5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        save_path  = st.text_input("Save model to", value="best_model.pth")
    with c2:
        lr           = st.number_input("Learning Rate", value=1e-3, format="%.5f", step=1e-4)
        weight_decay = st.number_input("Weight Decay",  value=1e-4, format="%.6f")
        patience     = st.slider("Early Stop Patience", 1, 10, 3)
    with c3:
        pretrained      = st.checkbox("ImageNet Pretrained", value=True)
        freeze_backbone = st.checkbox("Freeze Backbone", value=False)

    # ── Start training ────────────────────────────────────────
    if st.button("🚀  Start Training"):

        from core.model   import build_model
        from core.dataset import build_dataloaders
        from core.trainer import train as run_training

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Running on: **{device.upper()}**")

        image_dir = st.session_state["image_dir"]
        sensitive_cols = st.session_state["sensitive_cols"]
        outcome_col = st.session_state["outcome_col"]
        val_split = st.session_state.get("val_split", 0.2)

        with st.spinner("Building dataloaders…"):
            try:
                train_loader, val_loader, num_classes, label_map, df, group_indices = build_dataloaders(
                    image_dir=image_dir,
                    metadata_df=st.session_state["df_clean"],
                    image_paths=st.session_state.get("image_paths"),
                    sensitive_cols=sensitive_cols,
                    outcome_col=outcome_col,
                    batch_size=batch_size,
                    val_split=val_split,
                    stratify_by=sensitive_cols,
                    device=device,
                )
                st.session_state["df_image"] = df
                st.session_state["group_indices_image"] = group_indices
            except Exception as e:
                st.error(f"❌ Dataset loading failed: {e}")
                return

        st.success(f"✅ Dataset loaded — {num_classes} classes: {label_map}")
        model = build_model(num_classes, pretrained, freeze_backbone, device)
        st.success("✅ Model built — ResNet-18 + bias-aware head")

        prog_bar   = st.progress(0)
        status_box = st.empty()
        metric_box = st.empty()
        history    = {"train_loss": [], "val_loss": [], "val_acc": []}

        def progress_cb(epoch, total, tl, vl, va):
            prog_bar.progress(epoch / total)
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            metric_box.markdown(
                f"<span class='metric-pill'>Epoch {epoch}/{total}</span>"
                f"<span class='metric-pill warn'>TrainLoss {tl:.4f}</span>"
                f"<span class='metric-pill'>ValLoss {vl:.4f}</span>"
                f"<span class='metric-pill good'>ValAcc {va:.3f}</span>",
                unsafe_allow_html=True)

        def status_cb(msg):
            status_box.code(msg)

        try:
            final_history, best_acc = run_training(
                model, train_loader, val_loader,
                dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                     patience=patience, save_path=save_path),
                device,
                progress_callback=progress_cb,
                status_callback=status_cb,
            )
        except Exception as e:
            st.error(f"❌ Training error: {e}")
            return

        prog_bar.progress(1.0)
        st.success(f"✅ Training complete! Best Val Accuracy: **{best_acc:.4f}**")

        st.session_state["model"]       = model
        st.session_state["label_map"]   = label_map
        st.session_state["history"]     = final_history
        st.session_state["outcome_col"] = outcome_col
        st.session_state["val_loader"]  = val_loader
        st.session_state["device"]      = device
        st.session_state["num_classes"] = num_classes

        st.balloons()
        st.info("Model saved! Go to **XAI / Grad-CAM** to visualize predictions.")
