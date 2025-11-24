import streamlit as st
import pandas as pd
import sqlite3
import os
import time

st.markdown(
    """
    <style>
    
    .stApp {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1.5px solid #2f3136 !important;
        padding: 1rem 1rem !important;
    }

    div[data-baseweb="select"]:focus-within > div {
        border: 2px solid #3b82f6 !important;    /* BLUE inner border */
        box-shadow: 0 0 0 1px rgba(59,130,246,.2) !important;
    }

    div[data-baseweb="select"] .css-1n76uvr {
        color: inherit;
    }

    .stTextArea > div,
    [data-testid="stTextArea"] > div:first-child {
        border: none !important;                 
        border-radius: 10px !important;         
        background: transparent !important;
    }

    .stTextArea:focus-within > div,
    [data-testid="stTextArea"]:focus-within > div:first-child {
        border: 2px solid #3b82f6 !important;    /* BLUE inner border */
        box-shadow: 0 0 0 1px rgba(59,130,246,.2) !important;
    }

    .stTextArea:hover,
    [data-testid="stTextArea"]:hover {
        border-color: #5b6c7e !important;
    }

    .stTextArea textarea,
    [data-testid="stTextArea"] textarea {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        color: #ffffff !important;               
        font-size: 1rem !important;
        padding: 0.5rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
        resize: vertical !important;
        font-family: inherit !important;
    }

    /* Placeholder */
    .stTextArea textarea::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    
    .stDownloadButton>button, .stButton>button {
        background-color: #1a1c24 !important;   /* Discord-like dark gray */
        color: #ffffff !important;
        border: 1.5px solid #202229 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.25s ease !important;
        height: auto !important;
        width: 100% !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
    }

    .stDownloadButton>button:hover, .stButton>button:hover {
        background-color: #1d1f28 !important;
        border-color: #3b82f6 !important;       
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 15px rgba(88, 101, 242, 0.25) !important;
    }

    .stDownloadButton>button:active, .stButton>button:active {
        transform: translateY(0) !important;
        background-color: #32353b !important;
    }

    .stDownloadButton>button:focus-visible, .stButton>button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(88, 101, 242, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="SQLite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style="text-align:center; font-size:2.5rem;">
    Explore Model Training Datasets
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="text-align:center; font-size:1.1rem;">
    View and query the training data used to build predictive models for medical diagnoses.
    </p>
    """,
    unsafe_allow_html=True,
)

# Function to load CSV and convert it to SQLite
def csv_to_sqlite(csv_file):
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(":memory:")  
    df.to_sql('data', conn, index=False, if_exists='replace') 
    return conn, df

# Function to execute a query on the SQLite connection
def run_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        return str(e)

# Directory where CSVs are stored
csv_directory = './data'  

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

if len(csv_files) == 0:
    st.error("No CSV files found in the folder.")
else:
    # Select box to choose between CSVs
    selected_file = st.selectbox("Select CSV file", csv_files)

    # Load the selected CSV as DataFrame and SQLite connection
    csv_file_path = os.path.join(csv_directory, selected_file)
    conn, df = csv_to_sqlite(csv_file_path)

    # Pre-fill the query text area with an empty value 
    query = st.text_area("Enter SQL query (e.g., SELECT * FROM data WHERE column_name > 5):", 
                         value="", height=200)

    # Check if query is empty, if so, run the default query
    if query == "":
        # Default query if text box is empty
        query = "SELECT * FROM data LIMIT 10;"
        
    # Run Query with Timing
    start_time = time.time()
    result = run_query(conn, query)
    exec_time = (time.time() - start_time) * 1000

    # Display the query results
    if isinstance(result, pd.DataFrame):
        st.write("### Query Results:")
        st.dataframe(result)
        
        # Create two columns for buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Results as CSV",
                data=result.to_csv(index=False).encode(),
                file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("Copy Results to Clipboard", use_container_width=True):
                result.to_clipboard(index=False)
                
        st.caption(f"Query executed in **{exec_time:.1f} ms**")
    else:
        st.error(result)
        st.caption(f"Query executed in **{exec_time:.1f} ms**")
