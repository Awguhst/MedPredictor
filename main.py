import streamlit as st

st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(to right, #14171c, #1f2128);
    }

    [data-testid="stNavSectionHeader"] span {
        font-size: 17px !important;   
        font-weight: 700 !important;
    }
    
    .stApp {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

about_page = st.Page(
    "pages/about.py",
    title="About",
    icon=":material/account_circle:",
    default=True,
)
disease_page = st.Page(
    "pages/disease_prediction.py",
    title="Diagnostics",
    icon=":material/bar_chart:",
)

cardiovascular_page = st.Page(
    "pages/cardiovascular_predictor.py",
    title="Cardiovascular",
    icon=":material/favorite:",
)

sqlite_page = st.Page(
    "pages/sqlite.py",
    title="SQlite",
    icon=":material/database:",
)

pneumonia_page = st.Page(
    "pages/pneumonia.py",
    title="Pneumonia",
    icon=":material/air:",
)

tuberculosis_page = st.Page(
    "pages/tuberculosis.py",
    title="Tuberculosis",
    icon=":material/coronavirus:",
)

pg = st.navigation(
    {
        "Info": [about_page],
        "Tabular models": [disease_page, cardiovascular_page],
        "Vision models": [pneumonia_page,tuberculosis_page],
        "Data": [sqlite_page]
    }
)

pg.run()