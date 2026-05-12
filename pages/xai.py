"""
pages/xai.py ─ Grad-CAM / XAI Visualization page
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch


def render():
    st.markdown("<div class='bl-header' style='font-size:1.5rem'>🔬 XAI / Grad-CAM</div>", unsafe_allow_html=True)
    st.markdown("<div class='bl-sub'>Visual explanations — see exactly what the model 'looks at'</div>", unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️  Train or load a model first.")
        return

    model     = st.session_state["model"]
    label_map = st.session_state["label_map"]
    device    = st.session_state["device"]
    idx_to_lbl = {v: k for k, v in label_map.items()}

    # ── Settings ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        smooth   = st.checkbox("Smooth Grad-CAM (slower, cleaner)", value=False)
        alpha    = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5, 0.05)
    with c2:
        colormap_name = st.selectbox("Colormap", ["JET", "HOT", "RAINBOW", "PLASMA"])
        import cv2
        cmap_map = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "RAINBOW": cv2.COLORMAP_RAINBOW,
            "PLASMA": cv2.COLORMAP_PLASMA,
        }
        colormap = cmap_map[colormap_name]

    # ── Upload ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload image(s) — JPG or PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Upload one or more face images to generate explanations.")
        return

    from core.xai import GradCAM, SmoothGradCAM, pil_to_tensor, tensor_to_numpy_img

    CAMClass = SmoothGradCAM if smooth else GradCAM

    for uploaded_file in uploaded:
        pil_img  = Image.open(uploaded_file).convert("RGB")
        tensor   = pil_to_tensor(pil_img, device)

        cam_gen  = CAMClass(model)

        with torch.enable_grad():
            cam = cam_gen.generate(tensor, target_class=None)

        logits, embeddings = model(tensor)
        pred_idx   = logits.argmax(dim=1).item()
        pred_label = idx_to_lbl.get(pred_idx, str(pred_idx))
        confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()

        orig_np  = np.array(pil_img.resize((128, 128)))
        overlay  = cam_gen.overlay(cam, orig_np, alpha=alpha, colormap=colormap)

        # ── Display ───────────────────────────────────────────
        st.markdown(f"""
        <div class='bl-card'>
            <b>{uploaded_file.name}</b>
            <span class='metric-pill good' style='float:right'>
                {pred_label} — {confidence*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.image(orig_np,  caption="Original",  use_container_width=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.patch.set_facecolor("#0a0d14")
        col2.pyplot(fig, use_container_width=True)
        plt.close(fig)

        col3.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

        # ── Per-class activations ──────────────────────────────
        with st.expander("Per-class CAM analysis"):
            class_cols = st.columns(min(len(label_map), 4))
            for i, (cls_name, cls_idx) in enumerate(label_map.items()):
                cam_cls = cam_gen.generate(tensor, target_class=cls_idx)
                ov_cls  = cam_gen.overlay(cam_cls, orig_np, alpha=alpha, colormap=colormap)
                class_cols[i % len(class_cols)].image(
                    ov_cls, caption=f"Class: {cls_name}", use_container_width=True)

        st.divider()
