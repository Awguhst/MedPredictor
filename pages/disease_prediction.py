import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

if "shap_ready" not in st.session_state:
    st.session_state.shap_ready = False
    st.session_state.prediction_probs = None
    st.session_state.top_5_diseases = None
    st.session_state.top_5_indices = None
    st.session_state.shap_values = None
    st.session_state.expected_value = None
    st.session_state.input_df = None
    st.session_state.input_vector = None

# Load trained model
with open('./models/disease_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Load column names from model or a file
columns = list(model.feature_names_in_)

# Add this near the top after imports
st.set_page_config(
    page_title="Medical Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main font & colors */
    .main {
        background-color: #2c2f33;
    }
    .stApp {
        ffont-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 2.8rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 3.8rem;
        color: #b3b3b3;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 2px solid #23252b;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #2a56a4, #3d6de7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        margin-left: -6px; 
        margin-top: 10px; 
        transition: all 0.3s;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1.5px solid #2f3136;
        background: #11141a;
        min-height: 72px;
        padding: 1rem 1.5rem;
        margin-top: -40px;
    }

    div[data-baseweb="select"]:focus-within > div {
        border: 1px solid #3b82f6;    
        box-shadow: 0 0 0 1px rgba(59,130,246,.2);
    }

    div[data-baseweb="select"] .css-1n76uvr {
        color: inherit;
    }

    /* Background of a selected tag */
    div[data-baseweb="select"] [data-baseweb="tag"] {
        background: linear-gradient(to right, #2a56a4, #3d6de7);     
        color: #ffffff;           
    }

    /* Background when the tag is *hovered* */
    div[data-baseweb="select"] [data-baseweb="tag"]:hover {
        background: #1d4ed8;      
    }

    div[data-baseweb="select"] [data-baseweb="tag"] > span:last-child {
        color: #e0e7ff;          
    }

    div[data-baseweb="tag"] {
        border-radius: 2px;
    }

    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 12px;
        font-weight: 600;
    }

    .stAlert {
        border-radius: 12px;
    }
    
    details[class*="st-emotion-cache"] summary[class*="st-emotion-cache"] {
        background-color: #202229;
    }
    
    details[class*="st-emotion-cache"][open] summary[class*="st-emotion-cache"] {
        background-color: #262730;
    }
    
    button[data-testid="stTab"]
    div[data-testid="stMarkdownContainer"]
    p {
        font-family: 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Title for this page
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='title'>🩺 Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Leverage Machine Learning for Disease Prediction based on Symptom Data</p>", unsafe_allow_html=True)

with col2:
    # Force the multiselect to sit inside a column with the same width as the button
    selected_features = st.multiselect(
        "",
        options=columns,
        default=[],
        help="",
        key="symptoms_multiselect"
    )


if len(selected_features) >= 3:
    # PREPARE INPUT VECTOR
    input_vector = np.zeros(len(columns))
    for feature in selected_features:
        input_vector[columns.index(feature)] = 1
    input_df = pd.DataFrame([input_vector], columns=columns)

# PREDICT BUTTON
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    make_prediction_button = st.button(
        "**Predict Possible Diseases**",
        use_container_width=True,
        type="primary",
        key="predict_btn"
    )

# When the button is clicked
if make_prediction_button:
    if len(selected_features) >= 3:
        # Build the input vector
        input_vector = np.zeros(len(columns))
        for feature in selected_features:
            input_vector[columns.index(feature)] = 1
        input_df = pd.DataFrame([input_vector], columns=columns)

        # Predict probabilities
        prediction_probs = model.predict_proba(input_df)[0]

        # Top‑5 
        top_5_indices = np.argsort(prediction_probs)[::-1][:5]
        top_5_diseases = [model.classes_[i] for i in top_5_indices]
        top_5_probs = prediction_probs[top_5_indices]

        # SHAP
        background = np.zeros((1, len(columns)))
        background_df = pd.DataFrame(background, columns=columns)

        explainer = shap.LinearExplainer(model, background_df)
        shap_values = explainer(input_df)               
        expected_value = explainer.expected_value     

        # Cache everything
        st.session_state.update(
            shap_ready=True,
            prediction_probs=prediction_probs,
            top_5_diseases=top_5_diseases,
            top_5_indices=top_5_indices,
            top_5_probs=top_5_probs,
            shap_values=shap_values,
            expected_value=expected_value,
            input_df=input_df,
            input_vector=input_vector,
        )
    else:
        with col2:
            st.error("Please select at least 3 symptoms to make a prediction.")
        st.session_state.shap_ready = False
        

if st.session_state.shap_ready:
    st.markdown("##### Most Likely Diseases")

    # Define a function to categorize certainty
    def categorize_certainty(prob):
        if prob >= 0.75:
            return "High Certainty"
        elif prob >= 0.45:
            return "Medium Certainty"
        else:
            return "Low Certainty"

    # Prepare DataFrame
    disease_certainty_df = pd.DataFrame({
        'Disease': st.session_state.top_5_diseases,
        'Certainty': [categorize_certainty(p) for p in st.session_state.top_5_probs]
    })

    # Generate short explanations for all 5 diseases
    if "disease_explanations" not in st.session_state:
        try:
            # Load API key from secrets
            genai.configure(api_key=st.secrets["gemini_api_key"])
            gemini_api = genai.GenerativeModel("gemini-2.5-flash")

            diseases_list = "\n".join([f"- {d}" for d in st.session_state.top_5_diseases])

            prompt = f"""
                Give a very short (1 sentence, max 18 words) patient-friendly explanation for each condition below.
                Format exactly: Disease Name: explanation text
                No extra text, numbers, or quotes.

                Conditions:
                {diseases_list}
                """

            response = gemini_api.generate_content(prompt)
            raw_text = response.text.strip()

            explanations_dict = {}
            for line in raw_text.split('\n'):
                if ':' in line:
                    name_part, expl = line.split(':', 1)
                    disease_name = name_part.strip()
                    explanations_dict[disease_name] = expl.strip()

            st.session_state.disease_explanations = explanations_dict
        except Exception as e:
            # Fallback if API fails
            st.session_state.disease_explanations = {
                d: "A medical condition that requires professional evaluation."
                for d in st.session_state.top_5_diseases
            }

    # Create 5 columns for cards
    cols = st.columns(5)

    # Display each card with disease + short explanation + certainty
    for i, (disease, certainty) in enumerate(zip(disease_certainty_df['Disease'], disease_certainty_df['Certainty'])):
        with cols[i]:
            # Get explanation (with fallback)
            explanation = st.session_state.disease_explanations.get(
                disease,
                "Brief description unavailable."
            )

            st.markdown(
                f"""
                <div class="card" style="
                    background: linear-gradient(to right, #1a1d23, #262730);
                    padding: 20px;
                    height: 240px;  <!-- Increased to fit explanation -->
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    border-radius: 12px !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    border: 2px solid #23252b;
                    transition: all 0.3s ease;
                    ">
                    <div>
                        <h3 style="
                            text-align: center;
                            font-size: 18px;
                            font-weight: bold;
                            color: #ffffff;
                            margin: 0 0 12px 0;
                            ">{disease}</h3>
                        <p style="
                            text-align: center;
                            font-size: 14.5px;
                            color: #b0bec5;
                            line-height: 1.4;
                            margin: 0 0 20px 0;
                            ">{explanation}</p>
                    </div>
                    <p style="
                        text-align: center;
                        font-size: 16px;
                        color: #ffffff;
                        font-weight: 600;
                        margin: 0;
                        ">{certainty}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")
    
    # Confidence Dashboard
    if st.session_state.shap_ready:
        top_prob    = st.session_state.top_5_probs[0]
        top_disease = st.session_state.top_5_diseases[0]
        gap_pts     = (top_prob - st.session_state.top_5_probs[1]) * 100

        # Risk level 
        if top_prob > 0.75:
            certainty, badge_color = "High", "#f44e42"
        elif top_prob > 0.45:
            certainty, badge_color = "Moderate", "#ff9800"
        else:
            certainty, badge_color = "Low", "#4caf50"

        with st.container():
            st.markdown("<div class='conf-card'>", unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1], gap="medium")

            with col1:
                title_html = f"<b style='font-size:16px;'>{top_disease}</b>"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=top_prob*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': title_html, 'font': {'size': 16}},
                    delta={'reference': st.session_state.top_5_probs[1]*100,
                        'position': "top",
                        'increasing': {'color': "#4caf50"},
                        'decreasing': {'color': "#f44e42"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': badge_color},
                        'bgcolor': "#262730",
                        'borderwidth': 2,
                        'bordercolor': "#A8C8E0",
                        'steps': [
                            {'range': [0, 45], 'color': '#2e7d32'},
                            {'range': [45, 75], 'color': '#fb8c00'},
                            {'range': [75, 100], 'color': '#c62828'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': top_prob*100}
                    }))

                fig.update_layout(
                    paper_bgcolor="#202229",
                    font={'color': "#d0d0d0"},
                    height=200,                     
                    margin=dict(l=10, r=10, t=52, b=8)   
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False})

            with col2:
                # Badge
                st.markdown(
                    f"""
                    <div style="
                        background:#202229;
                        padding:10px;
                        border-radius:10px;
                        text-align:center;
                        border:2px solid {badge_color};
                        ">
                        <h4 style="margin:0; color:{badge_color};">{certainty} Certainty</h4>
                        <p style="margin:4px 0 0; color:#d0d0d0; font-size:0.85rem;">
                            +{gap_pts:.0f} pts vs #2
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                probs = [p*100 for p in st.session_state.top_5_probs]
                spark = go.Figure(go.Scatter(
                    y=probs,
                    mode='lines+markers',
                    line=dict(color='#4fc3f7', width=2),
                    marker=dict(size=5, color='#4fc3f7'),
                    hoverinfo='none'
                ))
                spark.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, range=[0, 100]),
                    height=60,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(spark, use_container_width=True,
                                config={'staticPlot': True})

            st.markdown("</div>", unsafe_allow_html=True)   
    
    st.markdown("")
        
    with st.expander("**Explainable AI: Why this prediction?** (SHAP Analysis)", expanded=False):
        
        st.markdown("""
        > **Select a disease** to analyze the prediction made by the model. The SHAP analysis will show how each feature contributes to the final decision, helping you understand why the model made a specific prediction.
        """)
        
        st.markdown("")

        selected_disease = st.selectbox(
            "Analyze prediction for:",
            options=st.session_state.top_5_diseases,
            format_func=lambda x: f"{x}",
            key="shap_selector"
        )

        class_idx = list(model.classes_).index(selected_disease)

        # Recreate explanation
        shap_vals_class = st.session_state.shap_values[0, :, class_idx]
        base_val = st.session_state.expected_value[class_idx]

        shap_explanation = shap.Explanation(
            values=shap_vals_class,
            base_values=base_val,
            data=st.session_state.input_vector,
            feature_names=columns,
        )

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
            shap.plots.waterfall(shap_explanation, max_display=MAX_DISPLAY, show=False)
            plt.subplots_adjust(**PAD_WATERFALL)
            st.pyplot(fig, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

        # Bar Tab
        with tab2:
            apply_bluish_background()
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            shap.plots.bar(shap_explanation, max_display=MAX_DISPLAY, show=False)
            plt.subplots_adjust(**PAD_BAR)
            st.pyplot(fig, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)