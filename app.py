"""
╔══════════════════════════════════════════════════════════════╗
║   BiasLens — Universal Dataset Bias Analyzer                 ║
║   Upload any CSV → instant fairness audit + mitigation       ║
║   Author  : Ananyaa P                                        ║
║   Roll No : 2127220501014                                    ║
║   Guide   : Dr. N Rajganesh                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
st.set_page_config(
    page_title="BiasLens — Universal Bias Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0d14;
    --surface:   #111520;
    --border:    #1e2535;
    --accent:    #4f8ef7;
    --accent2:   #f7614f;
    --accent3:   #4ff7b8;
    --text:      #e8ecf4;
    --muted:     #6b7592;
    --card:      #141926;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.bl-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}
.bl-card:hover { border-color: var(--accent); }
.metric-pill {
    display: inline-block;
    background: #1a2040;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: .4rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: .85rem;
    color: var(--accent);
    margin: .2rem;
}
.metric-pill.warn  { border-color: var(--accent2); color: var(--accent2); }
.metric-pill.good  { border-color: var(--accent3); color: var(--accent3); }
.bl-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #4f8ef7, #4ff7b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
}
.bl-sub { color: var(--muted); font-size: .95rem; margin-bottom: 2rem; }
.stProgress > div > div { background: var(--accent) !important; }
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .8rem !important;
    padding: .5rem 1.5rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }
hr { border-color: var(--border) !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 8px; }
[data-baseweb="tab"]      { color: var(--muted) !important; }
[aria-selected="true"]    { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Page router ────────────────────────────────────────────────
from pages import home, upload, upload_images, bias, dashboard, mitigation, train, xai

PAGES = {
    "🏠  Home":               home.render,
    "📂  Upload & Configure": upload.render,
    "🖼️   Upload Images":      upload_images.render,
    "📊  Bias Report":        bias.render,
    "📈  Dashboard":          dashboard.render,
    "🛠   Mitigation":        mitigation.render,
    "🧠  Train Model (Images)": train.render,
    "🔬  XAI / Grad-CAM":    xai.render,
}

with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:1.3rem;
                    background:linear-gradient(90deg,#4f8ef7,#4ff7b8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:700;letter-spacing:-1px'>BiasLens</div>
        <div style='color:#6b7592;font-size:.75rem;margin-top:.2rem'>
            Universal Dataset Bias Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show a green dot if dataset is loaded
    if st.session_state.get("upload_confirmed"):
        ds_name = st.session_state.get("dataset_name", "dataset")
        st.markdown(f"""
        <div style='background:#0d1f0d;border:1px solid #4ff7b8;border-radius:8px;
                    padding:.5rem .8rem;margin-bottom:1rem;font-size:.75rem'>
            <span style='color:#4ff7b8'>●</span>
            <span style='color:#6b7592;margin-left:.4rem'>{ds_name[:28]}</span>
        </div>
        """, unsafe_allow_html=True)

    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7592;font-size:.72rem;line-height:1.7'>
        <b style='color:#e8ecf4'>Ananyaa P</b><br>
        2127220501014<br>
        Guide: Dr. N Rajganesh<br>
        CS22811 – Project Work
    </div>
    """, unsafe_allow_html=True)

PAGES[page]()
