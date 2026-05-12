"""
pages/home.py — Landing page
"""
import streamlit as st


def render():
    st.markdown("""
    <div class='bl-header'>BiasLens 🔍</div>
    <div class='bl-sub'>Upload <i>any</i> dataset — instantly find, visualize, and fix bias in AI decisions</div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("📂", "Any CSV Dataset", "Upload any tabular dataset — hiring, loans, healthcare, criminal justice. No images required."),
        ("🤖", "Auto-Detection", "Automatically identifies sensitive attributes (gender, race, age) and outcome columns."),
        ("📊", "6 Fairness Metrics", "Bias Gap, Demographic Parity, Equalized Odds, Disparate Impact, Predictive Parity, Chi-square."),
        ("🛠", "Actionable Fixes", "Specific mitigation steps with before/after metric comparison — not just warnings."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class='bl-card' style='text-align:center'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-weight:600;margin:.5rem 0 .3rem'>{title}</div>
                <div style='color:#6b7592;font-size:.85rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### How to use BiasLens")
        steps = [
            ("1", "Upload Dataset",    "Go to **Upload & Configure** → upload any CSV file or pick a demo dataset."),
            ("2", "Confirm Columns",   "Review auto-detected sensitive & outcome columns, adjust if needed."),
            ("3", "Bias Report",       "See all 6 fairness metrics with plain-English verdicts per group."),
            ("4", "Mitigation",        "Apply reweighing, resampling, or threshold tuning — see updated metrics instantly."),
            ("5", "Image Mode",        "Optionally use **Train & XAI** for image datasets with Grad-CAM explanations."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class='bl-card' style='display:flex;gap:1rem;align-items:flex-start'>
                <div style='min-width:2rem;height:2rem;background:#1a2040;border:1px solid #4f8ef7;
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-family:Space Mono,monospace;color:#4f8ef7;font-size:.85rem'>{num}</div>
                <div>
                    <div style='font-weight:600'>{title}</div>
                    <div style='color:#6b7592;font-size:.85rem;margin-top:.2rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### Real-world datasets you can try")
        demos = [
            ("COMPAS Recidivism",    "ProPublica study — racial bias in US criminal risk scores"),
            ("Adult Income (UCI)",   "Census data — gender & race bias in income prediction"),
            ("German Credit",        "Loan approval — age & gender bias in credit decisions"),
            ("Your own CSV",         "Any dataset with a decision column + demographic columns"),
        ]
        for name, desc in demos:
            st.markdown(f"""
            <div class='bl-card' style='display:flex;gap:1rem;align-items:center'>
                <div style='color:#4f8ef7;font-size:1.2rem'>▸</div>
                <div>
                    <div style='font-weight:600;font-size:.9rem'>{name}</div>
                    <div style='color:#6b7592;font-size:.8rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='bl-card' style='border-color:#4ff7b8;margin-top:1rem'>
            <div style='color:#4ff7b8;font-size:.8rem;font-weight:600'>WHY THIS MATTERS</div>
            <div style='color:#6b7592;font-size:.82rem;margin-top:.4rem;line-height:1.6'>
                The EU AI Act (2024) legally requires bias audits for high-risk AI systems
                in hiring, credit, healthcare, and law enforcement.
                BiasLens generates the audit report automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
