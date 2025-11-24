import streamlit as st

# Page config
st.set_page_config(
    page_title="Medical Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

st.markdown("""
<style>
    .block-container {max-width: 850px; margin: auto;}
    
    /* Main app background & font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    /* title */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e40af, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #1a1d29, #212430);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        border: 1px solid #30363d;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(96, 165, 250, 0.25);
        border-color: #60a5fa;
    }
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }

    .disclaimer {
        background: rgba(251, 146, 60, 0.12);   /* subtle orange glow */
        border-left: 6px solid #fb923c;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    .disclaimer h3 {
        margin-top: 0;
        color: #fdba74;
        font-size: 1.3rem;
    }
    .disclaimer p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    /* General text adjustments */
    h4 {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Medical Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto; font-size: 1.15rem; color: #cbd5e1; line-height: 1.8;">
The <strong>Medical Predictor</strong> uses machine learning to analyze symptoms, clinical data, and medical images, providing instant, data-driven predictions. Explore its key features below.
</div>
""", unsafe_allow_html=True)

st.markdown("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Disease Predictor</div>
        <div class="feature-desc">Enter your symptoms and get instant predictions of possible conditions you may have.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Cardiovascular Risk</div>
        <div class="feature-desc">Predict your risk of heart disease with binary classification, and view feature importance through SHAP values and waterfall plots.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">X-Ray & Image Analysis</div>
        <div class="feature-desc">Upload chest X-rays for automated detection of pneumonia, tuberculosis, and other abnormalities.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Data Explorer</div>
        <div class="feature-desc">Select and query the different databases used for training the models, then view and export the results..</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <h3>⚠️ Important Medical Disclaimer</h3>
    <p>
        This tool is for <strong>educational and research purposes only</strong>.<br>
        All results are AI-generated and <strong>not a substitute</strong> for professional medical diagnosis or advice.<br><br>
        <strong>Always consult a qualified healthcare provider</strong> for any health concerns.
    </p>
</div>
""", unsafe_allow_html=True)