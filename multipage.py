# Streamlit app with two tabs
def main():
    import streamlit as st

    st.set_page_config(page_title="Two-Tab App", layout="wide")
    st.title("📊 Two-Tab Streamlit App")

    # Create two tabs at the top
    tab1, tab2 = st.tabs(["Overview", "Details"])

    # Tab 1: Overview
    with tab1:
        st.header("🔍 Overview")
        st.write("Welcome to the Overview tab. Here you might show summary metrics, charts, or a dashboard overview.")
        # Example: Display a placeholder metric
        metric_value = 12345
        st.metric(label="Total Visitors", value=f"{metric_value:,}")
        # Example: Simple line chart
        data = {'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], 'Sales': [100, 150, 120, 170, 200]}
        st.line_chart(data)

    # Tab 2: Details
    with tab2:
        st.header("📋 Details")
        st.write("Welcome to the Details tab. Here you can show detailed tables, logs, or configurations.")
        # Example: Data upload and preview
        uploaded_file = st.file_uploader("Upload CSV for Details", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("Please upload a CSV file to view details.")

if __name__ == "__main__":
    import pandas as pd
    main()
