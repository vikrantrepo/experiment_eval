import streamlit as st
import datetime

# -------------------- UI: Input Panel --------------------
st.title("SQL Query Builder")

with st.sidebar:
    st.header("Input Parameters")
    start_date = st.date_input("Reference Start Date", value=datetime.date(2025, 5, 5))
    end_date = st.date_input("Reference End Date", value=datetime.date(2025, 5, 14))
    date_time = st.text_input("Test Start Timestamp", value="2025-05-05 11:00:00")
    control_mvvar3 = st.text_input("Control mvvar3", value="%4747_new_login_registration_page^va:tru%")
    test_mvvar3 = st.text_input("Test mvvar3", value="%4747_new_login_registration_page^va:fal%")

    post_evar59 = st.multiselect("Shops (post_evar59)", options=[
        'zooplus.de', 'zooplus.pl', 'zooplus.fr', 'zooplus.it', 'zooplus.nl', 'zooplus.es',
        'zooplus.co.uk', 'zooplus.hu', 'zooplus.ro', 'zoohit.cz', 'zooplus.se', 'zooplus.be',
        'zooplus.ch', 'bitiba.de', 'zooplus.pt', 'zooplus.dk', 'zooplus.at', 'bitiba.pl',
        'zoohit.sk', 'zooplus.fi', 'bitiba.fr', 'bitiba.it', 'zooplus.no', 'bitiba.cz',
        'bitiba.es', 'bitiba.nl', 'zooplus.hr', 'zooplus.bg', 'zooplus.ie', 'zooplus.gr',
        'zoohit.si', 'zooplus.com', 'bitiba.co.uk', 'bitiba.se', 'bitiba.ch', 'bitiba.dk',
        'zoochic-eu.ru', 'bitiba.fi', 'bitiba.be', 'bitiba.com'
    ], default=['zooplus.de'])

    post_evar42 = st.multiselect("Devices (post_evar42)", options=["notApp", "ios", "android"], default=["notApp"])
    post_evar58 = st.text_input("URL Paths (comma-separated)", value="/checkout/login.htm, /checkout/login, /checkout/register")
    post_evar22 = st.text_input("Page Types (comma-separated)", value="checkout")

# -------------------- SQL Query Builder --------------------

def build_sql():
    shops = ", ".join(f"'{s}'" for s in post_evar59)
    devices = ", ".join(f"'{d}'" for d in post_evar42)

    url_filter_a = url_filter_b = ""
    if post_evar58.strip():
        urls = ", ".join(f"'{u.strip()}'" for u in post_evar58.split(","))
        url_filter_a = f"AND a.post_evar58 IN ({urls})"
        url_filter_b = f"AND b.post_evar58 IN ({urls})"

    page_filter_a = page_filter_b = ""
    if post_evar22.strip():
        pages = ", ".join(f"'{p.strip()}'" for p in post_evar22.split(","))
        page_filter_a = f"AND a.post_evar22 IN ({pages})"
        page_filter_b = f"AND b.post_evar22 IN ({pages})"

    timestamp_filter = f"AND date_time > timestamp '{date_time}'" if date_time else ""

    ref_start = start_date.strftime("%Y-%m-%d")
    ref_end = end_date.strftime("%Y-%m-%d")
    part_start = (start_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    part_end = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    query = f"""
WITH raw_exposures AS (
  SELECT 'Control' AS bucket,
        concat(post_visid_high, post_visid_low) AS visitor_id,
        MIN(date_time) AS first_exposure_timestamp
  FROM daci_privatespace.adobe_datafeeds a
  WHERE a.reference_date BETWEEN DATE '{ref_start}' AND DATE '{ref_end}'
    {timestamp_filter}
    AND a.post_mvvar3 LIKE '{control_mvvar3}'
    AND a.post_evar59 IN ({shops})
    AND a.post_evar42 IN ({devices})
    {url_filter_a}
    {page_filter_a}
  GROUP BY 1, 2
  UNION ALL
  SELECT 'Test' AS bucket,
        concat(post_visid_high, post_visid_low) AS visitor_id,
        MIN(date_time) AS first_exposure_timestamp
  FROM daci_privatespace.adobe_datafeeds b
  WHERE b.reference_date BETWEEN DATE '{ref_start}' AND DATE '{ref_end}'
    {timestamp_filter}
    AND b.post_mvvar3 LIKE '{test_mvvar3}'
    AND b.post_evar59 IN ({shops})
    AND b.post_evar42 IN ({devices})
    {url_filter_b}
    {page_filter_b}
  GROUP BY 1, 2
),
multi_bucket_visitors AS (
  SELECT visitor_id FROM raw_exposures GROUP BY visitor_id HAVING COUNT(DISTINCT bucket) = 1
),
bucketed_visitors_first_exposure AS (
  SELECT re.bucket AS buckets, re.visitor_id AS exposed_visitor_id, first_exposure_timestamp
  FROM raw_exposures re
  JOIN multi_bucket_visitors mbv ON re.visitor_id = mbv.visitor_id
),
visitor_cid AS (
  SELECT DISTINCT b.visitor_id AS mapped_visitor_id, b.customer_id AS mapped_customer_id
  FROM daci_datamarts.adobe_datafeeds_h b
  JOIN bucketed_visitors_first_exposure exp ON b.visitor_id = exp.exposed_visitor_id AND b.date_time >= exp.first_exposure_timestamp
  WHERE b.reference_date  BETWEEN DATE '{ref_start}' AND DATE '{ref_end}'
    {timestamp_filter}
    AND DATE(b.zoobrain_partition_key) between DATE '{part_start}' and DATE '{part_end}'
    AND b.shop_hostname IN ({shops})
    AND b.customer_id IS NOT NULL
),
orders_deduped AS (
  SELECT CAST(so.so_customer_id AS VARCHAR) AS customer_id,
        MIN(vc.mapped_visitor_id) AS attributed_visitor_id,
        so.so_id AS order_id,
        so.so_cm_net_sales_n AS net_sales,
        so.so_cm1_n AS cm1,
        so.so_cm2_n AS cm2,
        so.so_status_c AS order_status,
        so.so_order_counter_n AS order_counter,
        so.so_is_nc_order_f AS nc_order_f,
        so.so_is_tnc_order_f AS tnc_order_f,
        so.so_order_dt AS order_datetime
  FROM zoobrain_bo.sales_orders so
  JOIN visitor_cid vc ON vc.mapped_customer_id = CAST(so.so_customer_id AS VARCHAR)
  JOIN bucketed_visitors_first_exposure exp ON vc.mapped_visitor_id = exp.exposed_visitor_id
      AND so.so_order_dt > exp.first_exposure_timestamp
  GROUP BY 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
),
conversion_summary AS (
  SELECT attributed_visitor_id AS converted_visitor_id,
        order_id, net_sales, order_status, order_counter, cm1, cm2, nc_order_f, tnc_order_f
  FROM orders_deduped
)
SELECT exp.buckets,
      COUNT(DISTINCT exp.exposed_visitor_id) AS exposed_visitors,
      COUNT(DISTINCT cs.converted_visitor_id) AS converted_visitors_post_exposure,
      ROUND(COUNT(DISTINCT cs.converted_visitor_id) * 1.0 / COUNT(DISTINCT exp.exposed_visitor_id), 4) AS converted_visitor_share,
      ROUND(SUM(cs.net_sales)) AS total_net_sales,
      ROUND(SUM(cs.net_sales) / COUNT(DISTINCT exp.exposed_visitor_id), 2) AS net_sales_per_exposed_visitor,
      ROUND(SUM(cs.net_sales) / COUNT(DISTINCT CASE WHEN cs.order_status IN ('L','O') THEN cs.order_id ELSE NULL END), 2) AS net_AOV_with_L_O_orders,
      ROUND(SUM(cs.cm1) / COUNT(DISTINCT exp.exposed_visitor_id), 4) AS cm1_per_exposed_visitor,
      ROUND(SUM(cs.cm2) / COUNT(DISTINCT exp.exposed_visitor_id), 4) AS cm2_per_exposed_visitor,
      ROUND(SUM(cs.cm1) / SUM(cs.net_sales), 4) AS cm1_net_sales_share,
      ROUND(SUM(cs.cm2) / SUM(cs.net_sales), 4) AS cm2_net_sales_share,
      ROUND(COUNT(DISTINCT CASE WHEN cs.order_status IN ('L', 'O') THEN cs.order_id END) * 1.0 / NULLIF(COUNT(DISTINCT cs.converted_visitor_id), 0), 3) AS orders_per_converted_visitor_with_L_O_orders,
      COUNT(DISTINCT cs.order_id) AS orders,
      COUNT(DISTINCT CASE WHEN cs.order_status IN ('L','O') THEN cs.order_id ELSE NULL END) AS orders_L_O,
      ROUND(COUNT(DISTINCT CASE WHEN cs.order_status='S' THEN cs.order_id ELSE NULL END)*1.0 / COUNT(DISTINCT cs.order_id),5) AS orders_cancel_rate
FROM bucketed_visitors_first_exposure exp
LEFT JOIN conversion_summary cs ON exp.exposed_visitor_id = cs.converted_visitor_id
GROUP BY 1
ORDER BY 1
""".strip()
    return query

# -------------------- Output --------------------

st.subheader("Generated SQL Query")
sql_code = build_sql()
st.code(sql_code, language="sql")

