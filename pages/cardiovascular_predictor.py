import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

# Style
st.set_page_config(page_title="CardioRisk", layout="centered")

st.markdown(
    """
    <style>
    .main > div {padding-top: 2rem; padding-bottom: 2rem;}
    .block-container {max-width: 850px; margin: auto;}
    
    .stApp {
    background-color: #0e1117;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    h1, h2, h3, .stButton {text-align: center;}
    
    .stButton>button {
        background: linear-gradient(to right, #ff4b4b, #ff5959);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        margin-left: -6px; 
        margin-top: 15px; 
        transition: all 0.3s;
        box-shadow: 0 1px 3px rgba(255, 75, 75, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 6px rgba(255, 75, 75, 1);
    }
    
    .stSelectbox, .stNumberInput, .stCheckbox, .stTextInput {
        border: 1px solid #2f3136;
        border-radius: 6px;
        padding: 10px 20px;
        background: #11141a;
    }

    .stSelectbox:hover, .stNumberInput:hover, .stCheckbox:hover, .stTextInput:hover {
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    
    details[class*="st-emotion-cache"] summary[class*="st-emotion-cache"] {
        background-color: #171a21;
    }
    
    details[class*="st-emotion-cache"][open] summary[class*="st-emotion-cache"] {
        background-color: #262730;
    }
    
    button[data-testid="stTab"]
    div[data-testid="stMarkdownContainer"]
    p {
        font-family: 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .result-card {padding: 1.5rem; border-radius: 12px; text-align:center; margin-top: 35px;}
    .low-risk {background:#171a21; border:2px solid #00796b;}
    .high-risk {background:#171a21; border:2px solid #d32f2f;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Model
with open('./models/cardiovascular_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Title and description
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown(
    """
    <p style="text-align:center; font-size:1.1rem;">
    Enter the patient’s medical and lifestyle data. The model will instantly estimate the risk of cardiovascular disease.
    </p>
    """,
    unsafe_allow_html=True,
)

# User inputs
col_smoke, col_alco, col_active = st.columns(3)

with col_smoke:
    smoke_option = st.selectbox(
        "Smoking Status",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
        index=0,
        key="smoke_select"
    )
    smoke = smoke_option[1]

with col_alco:
    alco_option = st.selectbox(
        "Alcohol Consumption",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
        index=0,
        key="alco_select"
    )
    alco = alco_option[1]

with col_active:
    active_option = st.selectbox(
        "Regular Exercise",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
        index=1,  
        key="active_select"
    )
    active = active_option[1]

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=150,
        value=30,
    )
    gender = st.selectbox(
        "Gender", options=[(1, "Female"), (2, "Male")], format_func=lambda x: x[1]
    )[0]

with col2:
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=168)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=62)

col_bp1, col_bp2 = st.columns(2)
with col_bp1:
    ap_hi = st.number_input(
        "Systolic BP (mmHg)", min_value=0, max_value=300, value=110
    )
with col_bp2:
    ap_lo = st.number_input(
        "Diastolic BP (mmHg)", min_value=0, max_value=300, value=80
    )

col_chol, col_gluc = st.columns(2)
with col_chol:
    cholesterol = st.selectbox(
        "Cholesterol Level",
        options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
        format_func=lambda x: x[1],
    )[0]
with col_gluc:
    gluc = st.selectbox(
        "Glucose Level",
        options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
        format_func=lambda x: x[1],
    )[0]


col_btn1, col_btn2, col_btn3 = st.columns([1, 8, 1])
with col_btn2:
    predict_btn = st.button(
        "**Predict Possible Diseases**",
        use_container_width=True,
        type="primary",
        key="predict_btn"
    )

if predict_btn:
    # Build input DataFrame
    input_df = pd.DataFrame(
        {
            "age": [age*365],
            "gender": [gender],
            "height": [height],
            "weight": [weight],
            "ap_hi": [ap_hi],
            "ap_lo": [ap_lo],
            "cholesterol": [cholesterol],
            "gluc": [gluc],
            "smoke": [smoke],
            "alco": [alco],
            "active": [active],
        }
    )

    # Model inference
    risk = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    # Display result card
    if risk == 1:
        card_class = "high-risk"
        title = f"<span style='color: #d32f2f; font-size: 22px;'>High Susceptibility to Cardiovascular Disease</span>"  
        prob_text = f"<span style='color: #d32f2f; font-size: 20px;'>Probability: {prob[1]:.1%}</span>"  
    else:
        card_class = "low-risk"
        title = f"<span style='color: #00796b; font-size: 22px;'>Low Susceptibility to Cardiovascular Disease</span>"  
        prob_text = f"<span style='color: #00796b; font-size: 20px;'>Probability: {prob[0]:.1%}</span>" 

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <h3>{title}</h3>
            <p>{prob_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("")

    # SHAP Analysis
    background = np.zeros((1, input_df.shape[1]))
    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_df)

    # SHAP Waterfall and Bar Plots 
    with st.expander("**Explainable AI: Why this prediction?** (SHAP Analysis)", expanded=False):
        
        st.markdown("""
        > **SHAP analysis** helps explain how each factor (like age, blood pressure, cholesterol levels, etc.) influences the model’s prediction. It shows the positive or negative impact of each factor, making it easier to understand why the model thinks you may (or may not) be susceptible to a cardiovascular disease.
        """)
        
        # Tabs for different SHAP plots
        tab1, tab2 = st.tabs(["Waterfall", "Impact Bar"])

        MAX_DISPLAY = 7
        FIGSIZE = (3.2, 2.4)
        DPI = 90
        PAD_WATERFALL = dict(left=0.5, right=0.98, top=0.88, bottom=0.32)
        PAD_BAR = dict(left=0.52, right=0.98, top=0.88, bottom=0.28)

        # Theme
        BG_COLOR = "#262730"  
        FACE_COLOR = "#474955"  
        TEXT_COLOR = "#d0d0d0" 

        def apply_bluish_background():
            plt.rcParams.update({
                'figure.facecolor': BG_COLOR,
                'axes.facecolor': FACE_COLOR,
                'axes.labelcolor': TEXT_COLOR,
                'text.color': TEXT_COLOR,
                'xtick.color': TEXT_COLOR,
                'ytick.color': TEXT_COLOR,
                'axes.edgecolor': '#A8C8E0',
                'font.size': 8,
                'axes.titlesize': 9,
            })

        # Waterfall Tab
        with tab1:
            apply_bluish_background()
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            shap.waterfall_plot(shap_values[0])  
            plt.subplots_adjust(**PAD_WATERFALL)
            st.pyplot(fig, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

        # Bar Plot Tab
        with tab2:
            apply_bluish_background()
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            shap.plots.bar(shap_values)
            plt.subplots_adjust(**PAD_BAR)
            st.pyplot(fig, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
