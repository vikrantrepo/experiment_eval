import streamlit as st
import datetime
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Unified Experiment Tool", layout="wide")

# -------------------- SQL QUERY TAB --------------------
def sql_query_builder():
    st.title("SQL Query Builder")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Reference Start Date", value=datetime.date(2025, 5, 5))
        date_time = st.text_input("Test Start Timestamp", value="2025-05-05 11:00:00")
        control_mvvar3 = st.text_input("Control mvvar3", value="%4747_new_login_registration_page^va:tru%")
        post_evar59 = st.multiselect("Shops (post_evar59)", options=[
            'zooplus.de','zooplus.pl','zooplus.fr','zooplus.it','zooplus.nl','zooplus.es',
            'zooplus.co.uk','zooplus.hu','zooplus.ro','zoohit.cz','zooplus.se','zooplus.be',
            'zooplus.ch','bitiba.de','zooplus.pt','zooplus.dk','zooplus.at','bitiba.pl',
            'zoohit.sk','zooplus.fi','bitiba.fr','bitiba.it','zooplus.no','bitiba.cz',
            'bitiba.es','bitiba.nl','zooplus.hr','zooplus.bg','zooplus.ie','zooplus.gr',
            'zoohit.si','zooplus.com','bitiba.co.uk','bitiba.se','bitiba.ch','bitiba.dk',
            'zoochic-eu.ru','bitiba.fi','bitiba.be','bitiba.com'
        ], default=['zooplus.de'])
    with col2:
        end_date = st.date_input("Reference End Date", value=datetime.date(2025, 5, 14))
        test_mvvar3 = st.text_input("Test mvvar3", value="%4747_new_login_registration_page^va:fal%")
        post_evar42 = st.multiselect("Devices (post_evar42)", options=["notApp","ios","android"], default=["notApp"])
        post_evar58 = st.text_input("URL Paths (comma-separated)", value="/checkout/login.htm, /checkout/login, /checkout/register")
        post_evar22 = st.text_input("Page Types (comma-separated)", value="checkout")

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
"""
        return query

    st.subheader("Generated SQL Query")
    st.code(build_sql(), language="sql")

# -------------------- EXPERIMENT ANALYSIS HELPERS --------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'buckets', 'exposed_visitor_id', 'net_sales', 'order_id', 'order_status', 'device_platform', 'shop', 'cm1', 'cm2'}
    missing = required.difference(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df['net_sales'] = df['net_sales'].fillna(0.0)
    df['order_id'] = df['order_id'].fillna(0).astype(int)
    df['order_status'] = df['order_status'].fillna('Unknown').astype(str)
    df[['cm1', 'cm2']] = df[['cm1', 'cm2']].fillna(0.0)
    return df

def compute_bucket_metrics(grp: pd.core.groupby.DataFrameGroupBy) -> dict:
    total_visitors = grp['exposed_visitor_id'].nunique()
    converters = grp[(grp['order_id'] > 0) & grp['order_status'].isin(['L', 'O'])]['exposed_visitor_id'].nunique()
    orders_all = grp[grp['order_id'] > 0]['order_id'].nunique()
    orders_lo = grp[grp['order_status'].isin(['L', 'O'])]['order_id'].nunique()
    sales_sum = grp['net_sales'].sum()
    cancels = grp[grp['order_status'] == 'S']['order_id'].nunique()
    denom = orders_all if orders_all > 0 else None

    sum_cm1 = grp['cm1'].sum()
    sum_cm2 = grp['cm2'].sum()
    cm1_per_vis = sum_cm1 / total_visitors if total_visitors else 0
    cm2_per_vis = sum_cm2 / total_visitors if total_visitors else 0
    cm1_per_sales = sum_cm1 / sales_sum if sales_sum else 0
    cm2_per_sales = sum_cm2 / sales_sum if sales_sum else 0

    return {
        'total_visitors': total_visitors,
        'converting_visitors': converters,
        'conversion_rate': round(converters/total_visitors, 4) if total_visitors else 0,
        'orders_all': orders_all,
        'orders_L_O': orders_lo,
        'net_aov': round(sales_sum/orders_lo, 4) if orders_lo else 0,
        'orders_per_converting_visitor': round(orders_lo/converters, 4) if converters else 0,
        'share_of_cancelled_orders': round(cancels/denom, 4) if denom else 0,
        'net_sales_per_visitor': round(sales_sum/total_visitors, 4) if total_visitors else 0,
        'total_net_sales': round(sales_sum, 2),
        'cm1_per_total_visitors': cm1_per_vis,
        'cm2_per_total_visitors': cm2_per_vis,
        'cm1_per_total_net_sales': cm1_per_sales,
        'cm2_per_total_net_sales': cm2_per_sales
    }

def get_bucket_totals(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for bucket, grp in df.groupby('buckets'):
        rec = compute_bucket_metrics(grp)
        rec['bucket'] = bucket
        records.append(rec)
    totals = pd.DataFrame(records).set_index('bucket')
    return totals.reindex(['Control', 'Test'])

def compute_bucket_metrics_by_level(df, level):
    records = []
    for (lvl_val, bucket), grp in df.groupby([level, 'buckets']):
        rec = compute_bucket_metrics(grp)
        rec[level] = lvl_val
        rec['buckets'] = bucket
        records.append(rec)
    return pd.DataFrame(records).sort_values([level, 'buckets'])

def pivot_metrics(metrics_df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    df = metrics_df.set_index([index_col, 'buckets']).unstack('buckets')
    df.columns = [f"{metric}_{bucket}" for metric, bucket in df.columns]
    df = df.reset_index()
    df['conversion_rate_diff_bps'] = (df['conversion_rate_Test'] - df['conversion_rate_Control']) * 10000
    df['net_aov_rel_diff'] = ((df['net_aov_Test'] - df['net_aov_Control']) / df['net_aov_Control']).replace([np.inf, -np.inf], np.nan)
    df['orders_per_converter_rel_diff'] = ((df['orders_per_converting_visitor_Test'] - df['orders_per_converting_visitor_Control']) / df['orders_per_converting_visitor_Control']).replace([np.inf, -np.inf], np.nan)
    df['net_sales_per_visitor_abs_diff'] = df['net_sales_per_visitor_Test'] - df['net_sales_per_visitor_Control']
    df['net_sales_per_visitor_rel_diff'] = (df['net_sales_per_visitor_abs_diff'] / df['net_sales_per_visitor_Control']).replace([np.inf, -np.inf], np.nan)
    return df.round({
        'conversion_rate_diff_bps': 0,
        'net_aov_rel_diff': 4,
        'orders_per_converter_rel_diff': 4,
        'net_sales_per_visitor_abs_diff': 4,
        'net_sales_per_visitor_rel_diff': 4
    })

def bootstrap_rpev(df: pd.DataFrame, n_iters=10000, seed=42):
    visitor_sales = df.groupby(['buckets', 'exposed_visitor_id'], as_index=False)['net_sales'].sum()
    test = visitor_sales.loc[visitor_sales.buckets == 'Test', 'net_sales'].values
    ctrl = visitor_sales.loc[visitor_sales.buckets == 'Control', 'net_sales'].values
    obs = test.mean() - ctrl.mean()
    rng = np.random.default_rng(seed)
    diffs = np.array([
        rng.choice(test, size=len(test), replace=True).mean() -
        rng.choice(ctrl, size=len(ctrl), replace=True).mean()
        for _ in range(n_iters)
    ])
    p_val = np.mean(np.abs(diffs) >= abs(obs))
    ci = np.percentile(diffs, [2.5, 97.5])
    return obs, p_val, ci, diffs

def conversion_z_test(df: pd.DataFrame, alpha=0.05):
    df['converted'] = df['order_id'] > 0
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    successes = np.array([summary.loc['Test', 'sum'], summary.loc['Control', 'sum']])
    nobs = np.array([summary.loc['Test', 'count'], summary.loc['Control', 'count']])
    p_pool = successes.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1]))
    _, p = proportions_ztest(successes, nobs)
    z_alpha = norm.ppf(1 - alpha/2)
    diff = successes[0]/nobs[0] - successes[1]/nobs[1]
    ci = (diff - z_alpha * se, diff + z_alpha * se)
    return diff/se, p, ci

def mann_whitney_tests(df: pd.DataFrame):
    df_lo = df[df['order_status'].isin(['L', 'O'])]
    visitor = df_lo.groupby(['buckets', 'exposed_visitor_id']).agg(
        total_sales=('net_sales', 'sum'),
        order_count=('order_id', 'nunique')
    ).assign(
        net_aov=lambda x: x.total_sales / x.order_count,
        orders_per_converted=lambda x: x.order_count
    ).reset_index()
    t_o = visitor.loc[visitor.buckets == 'Test', 'orders_per_converted']
    c_o = visitor.loc[visitor.buckets == 'Control', 'orders_per_converted']
    t_a = visitor.loc[visitor.buckets == 'Test', 'net_aov']
    c_a = visitor.loc[visitor.buckets == 'Control', 'net_aov']
    u_o, p_o = mannwhitneyu(t_o, c_o, alternative='two-sided')
    u_a, p_a = mannwhitneyu(t_a, c_a, alternative='two-sided')
    return (u_o, p_o), (u_a, p_a)

def show_visuals(df: pd.DataFrame, index_col: str):
    cols = ['conversion_rate_diff_bps', 'net_sales_per_visitor_abs_diff', 'net_aov_rel_diff', 'orders_per_converter_rel_diff']
    sorted_df = df.sort_values(f'total_visitors_Test', ascending=False)
    for col in cols:
        if col in sorted_df.columns:
            st.write(f"**{col.replace('_', ' ').title()}**")
            base = alt.Chart(sorted_df).encode(
                x=alt.X(
                    index_col,
                    sort=list(sorted_df[index_col]),
                    axis=alt.Axis(labelAngle=-45, labelAlign='right', labelLimit=200)
                ),
                y=alt.Y(
                    col,
                    title=col.replace('_', ' ').title(),
                    axis=alt_AXIS(labelAngle=0, labelAlign='right', titlePadding=10)
                ),
                tooltip=[index_col, col]
            )
            bars = base.mark_bar()
            fmt = ".0f" if col == 'conversion_rate_diff_bps' else (".1%" if col in ['net_aov_rel_diff', 'orders_per_converter_rel_diff'] else ".2f")
            text = base.mark_text(
                align='center', baseline='bottom', dy=-4, fontSize=12
            ).encode(
                text=alt.Text(col, format=fmt),
                color=alt.condition(alt.datum[col] < 0, alt.value("red"), alt.value("green"))
            )
            chart = (bars + text).properties(
                height=300,
                width={'step':80}
            ).configure_axisLeft(
                labelAngle=0,
                labelAlign='right',
                titlePadding=10
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(chart, use_container_width=True)

# -------------------- EXPERIMENT DASHBOARD TAB --------------------
def experiment_dashboard():
    st.title("📊 Experiment Results")
    path = st.file_uploader("Upload CSV", type='csv')
    if not path:
        st.info("Please upload your experiment CSV file.")
        return
    df = load_and_clean(path)

    # Filters
    with st.expander("🔍 Filter Options", expanded=False):
        shops = sorted(df['shop'].unique())
        devs = sorted(df['device_platform'].unique())
        sel_shops = st.multiselect("Shops", shops, default=shops)
        sel_devs = st.multiselect("Devices", devs, default=devs)
    df = df[df['shop'].isin(sel_shops) & df['device_platform'].isin(sel_devs)]

    # Outlier removal
    df_lo_overall = df[df['order_status'].isin(['L', 'O'])]
    visitor_stats_all = df_lo_overall.groupby('exposed_visitor_id').agg(
        total_sales=('net_sales', 'sum'),
        order_count=('order_id', 'nunique')
    ).assign(
        net_aov=lambda x: x.total_sales / x.order_count,
        orders_per_converted=lambda x: x.order_count
    )
    aov_cutoff = visitor_stats_all['net_aov'].quantile(0.999)
    opc_cutoff = visitor_stats_all['orders_per_converted'].quantile(0.999)
    outlier_ids = visitor_stats_all.loc[
        (visitor_stats_all['net_aov'] > aov_cutoff) |
        (visitor_stats_all['orders_per_converted'] > opc_cutoff)
    ].index
    df = df[~df['exposed_visitor_id'].isin(outlier_ids)]

    excluded_counts = df_lo_overall[df_lo_overall['exposed_visitor_id'].isin(outlier_ids)].groupby('buckets')['exposed_visitor_id'].nunique().reindex(['Control','Test'], fill_value=0)
    st.write(f"**Excluded Visitors:** Control: {excluded_counts.loc['Control']}, Test: {excluded_counts.loc['Test']} (above 99.9th percentile)")

    # Overall metrics
    st.subheader("🏁 Overall Metrics by Bucket")
    totals_df = get_bucket_totals(df)
    ...
