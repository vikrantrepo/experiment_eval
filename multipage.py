# app.py
import streamlit as st
from datetime import date, timedelta
import pandas as pd

st.set_page_config(page_title="SQL Query Builder", layout="wide")
st.title("🛠️ SQL Query Builder")

# --- Two-column layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Filters")
    start_date = st.date_input(
        "Reference Start Date",
        value=date(2025, 5, 5),
    )
    end_date = st.date_input(
        "Reference End Date",
        value=date(2025, 5, 14),
    )
    test_timestamp = st.text_input(
        "Test Start Timestamp",
        value="2025-05-05 11:00:00",
    )
    control_mvvar3 = st.text_input(
        "Control mvvar3",
        value="%4747_new_login_registration_page^va:tru%",
    )
    test_mvvar3 = st.text_input(
        "Test mvvar3",
        value="%4747_new_login_registration_page^va:fal%",
    )
    shops = st.multiselect(
        "Shops (post_evar59)",
        options=[
            'zooplus.de', 'zooplus.pl', 'zooplus.fr', 'zooplus.it',
            'zooplus.nl', 'zooplus.es', 'zooplus.co.uk', 'zooplus.hu',
            'zooplus.ro', 'zoohit.cz', 'zooplus.se', 'zooplus.be',
            'zooplus.ch', 'bitiba.de', 'zooplus.pt', 'zooplus.dk',
            'zooplus.at', 'bitiba.pl', 'zoohit.sk', 'zooplus.fi',
            'bitiba.fr', 'bitiba.it', 'zooplus.no', 'bitiba.cz',
            'bitiba.es', 'bitiba.nl', 'zooplus.hr', 'zooplus.bg',
            'zooplus.ie', 'zooplus.gr', 'zoohit.si', 'zooplus.com',
            'bitiba.co.uk', 'bitiba.se', 'bitiba.ch', 'bitiba.dk',
            'zoochic-eu.ru', 'bitiba.fi', 'bitiba.be', 'bitiba.com'
        ],
        default=["zooplus.de"],
    )
    devices = st.multiselect(
        "Devices (post_evar42)",
        options=["notApp", "ios", "android"],
        default=["notApp"],
    )
    url_paths = st.text_input(
        "URL Paths (comma-separated)",
        value="/checkout/login.htm, /checkout/login, /checkout/register",
    )
    page_types = st.text_input(
        "Page Types (comma-separated)",
        value="checkout",
    )

with col2:
    st.subheader("Generated SQL")
    # Build SQL string
    shops_list = ", ".join(f"'{s}'" for s in shops) or "''"
    dev_list = ", ".join(f"'{d}'" for d in devices) or "''"

    url_clause = ""
    if url_paths:
        urls = ", ".join(f"'{u.strip()}'" for u in url_paths.split(","))
        url_clause = f"AND a.post_evar58 IN ({urls})\n      AND b.post_evar58 IN ({urls})"

    page_clause = ""
    if page_types:
        pages = ", ".join(f"'{p.strip()}'" for p in page_types.split(","))
        page_clause = f"AND a.post_evar22 IN ({pages})\n      AND b.post_evar22 IN ({pages})"

    ts_clause = ""
    if test_timestamp:
        ts_clause = f"AND date_time > timestamp '{test_timestamp}'"

    ref_start = start_date.strftime("%Y-%m-%d")
    ref_end = end_date.strftime("%Y-%m-%d")
    part_start = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")
    part_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    sql = f"""
WITH raw_exposures AS (
  SELECT 'Control' AS bucket,
         concat(post_visid_high, post_visid_low) AS visitor_id,
         MIN(date_time) AS first_exposure_timestamp
  FROM daci_privatespace.adobe_datafeeds a
  WHERE a.reference_date BETWEEN DATE '{ref_start}' AND DATE '{ref_end}'
    {ts_clause}
    AND a.post_mvvar3 LIKE '{control_mvvar3}'
    AND a.post_evar59 IN ({shops_list})
    AND a.post_evar42 IN ({dev_list})
    {url_clause}
    {page_clause}
  GROUP BY 1,2
  UNION ALL
  SELECT 'Test' AS bucket,
         concat(post_visid_high, post_visid_low) AS visitor_id,
         MIN(date_time) AS first_exposure_timestamp
  FROM daci_privatespace.adobe_datafeeds b
  WHERE b.reference_date BETWEEN DATE '{ref_start}' AND DATE '{ref_end}'
    {ts_clause}
    AND b.post_mvvar3 LIKE '{test_mvvar3}'
    AND b.post_evar59 IN ({shops_list})
    AND b.post_evar42 IN ({dev_list})
    {url_clause}
    {page_clause}
  GROUP BY 1,2
),
-- ... rest of your CTEs & final SELECT ...
"""
    # Render with a copy button
    import streamlit.components.v1 as components
    html = f"""
    <button id="copy-btn">Copy SQL</button>
    <pre id="sql-box" style="white-space: pre-wrap; font-family: monospace; font-size:13px;">{sql}</pre>
    <script>
    const btn = document.getElementById('copy-btn');
    btn.onclick = () => {{
      const text = document.getElementById('sql-box').innerText;
      navigator.clipboard.writeText(text).then(() => {{
        alert('SQL copied to clipboard!');
      }});
    }};
    </script>
    """
    components.html(html, height=300)
