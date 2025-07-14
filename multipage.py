# File: Home.py
import streamlit as st

st.set_page_config(page_title="Home", layout="wide")
st.title("📊 Experiment Results — Home")

st.write("Welcome to the Experiment Dashboard. Use the menu on the left to navigate between pages.")

# Upload and data loading
path = st.file_uploader("Upload CSV", type='csv')
if path:
    df = ...  # load and clean your DataFrame here
    st.write("Data preview:")
    st.dataframe(df.head(), use_container_width=True)

# Insight or overview
st.subheader("🏁 Overall Metrics by Bucket")
st.write("*(Placeholder for overall metrics table and insight.)*")


# File: /pages/1_Statistical_Tests.py
import streamlit as st

st.set_page_config(page_title="Statistical Tests", layout="wide")
st.title("🔬 Statistical Tests")

st.write("This page shows statistical test results for your experiment data.")

# Assume `df` is passed or reloaded here
# Example placeholder table
tests_df = st.session_state.get('tests_df', None)
if tests_df is not None:
    st.table(tests_df)
else:
    st.info("Upload data on the Home page to see test results.")

# File: /pages/2_Segment_Analysis.py
import streamlit as st

st.set_page_config(page_title="Segment Analysis", layout="wide")
st.title("📊 Segment Analysis")

st.write("This page shows segment-level impact insights.")

segments_df = st.session_state.get('segments_df', None)
if segments_df is not None:
    st.dataframe(segments_df)
else:
    st.info("Upload data on the Home page to see segment analysis.")
