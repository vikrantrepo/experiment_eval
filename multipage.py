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
    diff = pd.Series(index=totals_df.columns, name='Ab. Delta')
    cr_test = totals_df.loc['Test', 'conversion_rate']
    cr_ctrl = totals_df.loc['Control', 'conversion_rate']
    diff['conversion_rate'] = f"{int(round((cr_test - cr_ctrl) * 10000, 0))} bps"
    diff['net_aov'] = round(totals_df.loc['Test','net_aov'] - totals_df.loc['Control','net_aov'], 4)
    diff['orders_per_converting_visitor'] = round(totals_df.loc['Test','orders_per_converting_visitor'] - totals_df.loc['Control','orders_per_converting_visitor'], 4)
    diff['net_sales_per_visitor'] = round(totals_df.loc['Test','net_sales_per_visitor'] - totals_df.loc['Control','net_sales_per_visitor'], 4)
    diff['cm1_per_total_visitors'] = round(totals_df.loc['Test','cm1_per_total_visitors'] - totals_df.loc['Control','cm1_per_total_visitors'], 4)
    diff['cm2_per_total_visitors'] = round(totals_df.loc['Test','cm2_per_total_visitors'] - totals_df.loc['Control','cm2_per_total_visitors'], 4)
    for m in [ 'cm1_per_total_net_sales', 'cm2_per_total_net_sales']:
        test_v = totals_df.loc['Test', m]
        ctrl_v = totals_df.loc['Control', m]
        diff[m] = f"{abs(round((test_v - ctrl_v)*100, 2))}%"
    totals_with_diff = totals_df.copy()
    totals_with_diff.loc['Absolute Difference'] = diff
    color_metrics = [
        'conversion_rate', 'net_aov', 'orders_per_converting_visitor', 'net_sales_per_visitor',
        'cm1_per_total_visitors', 'cm2_per_total_visitors', 'cm1_per_total_net_sales', 'cm2_per_total_net_sales'
    ]
    def highlight_metric(col):
        vals = col.loc[['Control','Test']]
        max_val = vals.max()
        min_val = vals.min()
        return [
            ('background-color: lightgreen' if (idx in ['Control','Test'] and v==max_val)
             else 'background-color: salmon' if (idx in ['Control','Test'] and v==min_val)
             else '')
            for idx, v in col.items()
        ]
    styled = totals_with_diff.style
    for metric in color_metrics:
        styled = styled.apply(highlight_metric, subset=[metric], axis=0)
    fmt_dict = {
        'total_visitors': '{:,.0f}',
        'converting_visitors': '{:,.0f}',
        'orders_all': '{:,.0f}',
        'orders_L_O': '{:,.0f}',
        'total_net_sales': '€{:,.0f}',
        'conversion_rate': lambda v: f"{v:.2%}" if isinstance(v, (int, float, np.floating)) else v,
        'net_aov': lambda v: f"€{v:.2f}",
        'orders_per_converting_visitor': '{:.4f}',
        'share_of_cancelled_orders': '{:.2%}',
        'net_sales_per_visitor': lambda v: f"€{v:.2f}" if isinstance(v, (int, float, np.floating)) else v,
        'cm1_per_total_visitors': lambda v: f"€{v:.2f}" if isinstance(v, (int, float, np.floating)) else v,
        'cm2_per_total_visitors': lambda v: f"€{v:.2f}" if isinstance(v, (int, float, np.floating)) else v,
        'cm1_per_total_net_sales': lambda v: f"{v:.2%}" if isinstance(v, (int, float, np.floating)) else v,
        'cm2_per_total_net_sales': lambda v: f"{v:.2%}" if isinstance(v, (int, float, np.floating)) else v
    }
    styled = styled.format(fmt_dict)
    st.dataframe(styled, use_container_width=True)

    # -------------------- STATISTICAL TESTS & VISUALS --------------------
    obs, p_boot, ci_boot, diffs = bootstrap_rpev(df)
    z, p_z, ci_z = conversion_z_test(df)
    (u_o, p_o), (u_a, p_a) = mann_whitney_tests(df)
    stats_summary = pd.DataFrame([
        { 'Test': 'Revenue per Visitor (Bootstrap)', 'Statistic': f"{obs:.4f}", 'P-value': p_boot, 'CI Lower': ci_boot[0], 'CI Upper': ci_boot[1], 'Significant': 'Yes' if p_boot < 0.05 else 'No' },
        { 'Test': 'Conversion Rate (Z-test)', 'Statistic': f"{z:.4f}", 'P-value': p_z, 'CI Lower': ci_z[0], 'CI Upper': ci_z[1], 'Significant': 'Yes' if p_z < 0.05 else 'No' },
        { 'Test': 'Orders per Converter (Mann-Whitney)', 'Statistic': f"{u_o:.2f}", 'P-value': p_o, 'CI Lower': np.nan, 'CI Upper': np.nan, 'Significant': 'Yes' if p_o < 0.05 else 'No' },
        { 'Test': 'Net AOV (Mann-Whitney)', 'Statistic': f"{u_a:.2f}", 'P-value': p_a, 'CI Lower': np.nan, 'CI Upper': np.nan, 'Significant': 'Yes' if p_a < 0.05 else 'No' }
    ])
    total_vis_test = totals_df.loc['Test','total_visitors']
    cr_c = totals_df.loc['Control','conversion_rate']
    opc_c = totals_df.loc['Control','orders_per_converting_visitor']
    aov_c = totals_df.loc['Control','net_aov']
    delta_nspv = totals_df.loc['Test','net_sales_per_visitor'] - totals_df.loc['Control','net_sales_per_visitor']
    delta_cr = totals_df.loc['Test','conversion_rate'] - cr_c
    delta_opc = totals_df.loc['Test','orders_per_converting_visitor'] - opc_c
    delta_aov = totals_df.loc['Test','net_aov'] - aov_c
    net_sales_impact = delta_nspv * total_vis_test
    contr_cr = delta_cr * opc_c * aov_c * total_vis_test
    contr_opc = cr_c * delta_opc * aov_c * total_vis_test
    contr_aov = cr_c * opc_c * delta_aov * total_vis_test
    contributors = {'Conversion Rate': contr_cr, 'Orders per Converted Visitor': contr_opc, 'Net AOV': contr_aov}
    primary = max(contributors, key=lambda k: contributors[k]) if net_sales_impact >= 0 else min(contributors, key=lambda k: contributors[k])
    sign = 'positive' if net_sales_impact >= 0 else 'negative'

    # --- new conditional Insight ---
    alpha = 0.05
    if p_boot < alpha:
        # RPV is significant
        st.write(f"**Insight:** The difference in **Revenue per Visitor** is statistically significant _(p = {p_boot:.3f})_.")
        st.write(f"> Overall net-sales impact is **{sign}** (€{net_sales_impact:.2f}) and the primary driver is **{primary}**.")
    else:
        # RPV not significant: one paragraph with significance flags
        ctrl_nspv = totals_df.loc['Control','net_sales_per_visitor']
        test_nspv = totals_df.loc['Test','net_sales_per_visitor']
        ctrl_cr    = totals_df.loc['Control','conversion_rate']
        test_cr    = totals_df.loc['Test','conversion_rate']
        ctrl_aov   = totals_df.loc['Control','net_aov']
        test_aov   = totals_df.loc['Test','net_aov']
        ctrl_cm1v  = totals_df.loc['Control','cm1_per_total_visitors']
        test_cm1v  = totals_df.loc['Test','cm1_per_total_visitors']
        ctrl_cm2v  = totals_df.loc['Control','cm2_per_total_visitors']
        test_cm2v  = totals_df.loc['Test','cm2_per_total_visitors']
        ctrl_cm1s  = totals_df.loc['Control','cm1_per_total_net_sales']
        test_cm1s  = totals_df.loc['Test','cm1_per_total_net_sales']
        ctrl_cm2s  = totals_df.loc['Control','cm2_per_total_net_sales']
        test_cm2s  = totals_df.loc['Test','cm2_per_total_net_sales']

        sig_rpv = "significant" if p_boot < alpha else "not significant"
        sig_cr  = "significant" if p_z    < alpha else "not significant"
        sig_aov = "significant" if p_a    < alpha else "not significant"

        paragraph = (
            f"Revenue per visitor changed from €{ctrl_nspv:.2f} to €{test_nspv:.2f} "
            f"(p = {p_boot:.3f}, {sig_rpv}); conversion rate from {ctrl_cr:.2%} to {test_cr:.2%} "
            f"(p = {p_z:.3f}, {sig_cr}); net AOV from €{ctrl_aov:.2f} to €{test_aov:.2f} "
            f"(p = {p_a:.3f}, {sig_aov}). CM1 per visitor moved from €{ctrl_cm1v:.2f} to €{test_cm1v:.2f} "
            f"(no test), and CM2 per visitor from €{ctrl_cm2v:.2f} to €{test_cm2v:.2f} (no test). "
            f"By net sales, CM1 went from {ctrl_cm1s:.2%} to {test_cm1s:.2%}, and CM2 from "
            f"{ctrl_cm2s:.2%} to {test_cm2s:.2%} (no tests on CM metrics)."
        )
        st.write(f"**Insight:** {paragraph}")
    stats_summary['Impact'] = [net_sales_impact, contr_cr, contr_opc, contr_aov]
    st.subheader("🔬 Statistical Tests Summary")
    st.table(stats_summary.set_index('Test'))

    st.subheader("📈 Distribution and Boxplots")
    df_lo = df[df['order_status'].isin(['L', 'O'])]
    visitor_stats = df_lo.groupby(['buckets', 'exposed_visitor_id']).agg(
        total_sales=('net_sales', 'sum'),
        order_count=('order_id', 'nunique')
    ).assign(
        net_aov=lambda x: x.total_sales / x.order_count,
        orders_per_converted=lambda x: x.order_count
    ).reset_index()
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.hist(diffs, bins=50, alpha=0.7)
        ax1.axvline(obs, linestyle='--')
        ax1.axvline(ci_boot[0], linestyle=':')
        ax1.axvline(ci_boot[1], linestyle=':')
        ax1.set_title('Bootstrap Distribution')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        visitor_stats.boxplot(column='net_aov', by='buckets', ax=ax2)
        ax2.set_title('Net AOV by Bucket')
        ax2.set_xlabel('')
        ax2.set_ylabel('Net AOV')
        plt.suptitle('')
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        visitor_stats.boxplot(column='order_count', by='buckets', ax=ax3)
        ax3.set_title('Orders per Converted Visitor')
        ax3.set_xlabel('')
        ax3.set_ylabel('Orders per Visitor')
        plt.suptitle('')
        st.pyplot(fig3)

    shop_metrics = compute_bucket_metrics_by_level(df, 'shop')
    device_metrics = compute_bucket_metrics_by_level(df, 'device_platform')
    shop_pivot = pivot_metrics(shop_metrics, 'shop').sort_values('total_visitors_Test', ascending=False)
    device_pivot = pivot_metrics(device_metrics, 'device_platform').sort_values('total_visitors_Test', ascending=False)

    st.subheader("🛒 Shop-Level Metrics")
    st.dataframe(shop_pivot.reset_index(drop=True), use_container_width=True)

    st.subheader("📱 Device-Level Metrics")
    st.dataframe(device_pivot.reset_index(drop=True), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Shop-Level Visuals")
        show_visuals(shop_pivot, 'shop')
    with col2:
        st.subheader("📊 Device-Level Visuals")
        show_visuals(device_pivot, 'device_platform')

    def compute_contribs(df, segment_col):
        df = df.copy()
        df['net_sales_impact'] = df['net_sales_per_visitor_abs_diff'] * df[f'total_visitors_Test']
        df['cr_c'] = df[f'conversion_rate_Control']
        df['opc_c'] = df[f'orders_per_converting_visitor_Control']
        df['aov_c'] = df[f'net_aov_Control']
        df['delta_cr'] = df[f'conversion_rate_Test'] - df[f'conversion_rate_Control']
        df['delta_opc'] = df[f'orders_per_converting_visitor_Test'] - df[f'orders_per_converting_visitor_Control']
        df['delta_aov'] = df[f'net_aov_Test'] - df[f'net_aov_Control']
        df['contr_cr'] = df['delta_cr'] * df['opc_c'] * df['aov_c'] * df[f'total_visitors_Test']
        df['contr_opc'] = df['cr_c'] * df['delta_opc'] * df['aov_c'] * df[f'total_visitors_Test']
        df['contr_aov'] = df['cr_c'] * df['opc_c'] * df['delta_aov'] * df[f'total_visitors_Test']
        df['main_contributor'] = df.apply(lambda row: max({'Conversion Rate': row['contr_cr'], 'Orders per Converted Visitor': row['contr_opc'], 'Net AOV': row['contr_aov']}.items(), key=lambda x: x[1])[0] if row['net_sales_impact'] >= 0 else min({'Conversion Rate': row['contr_cr'], 'Orders per Converted Visitor': row['contr_opc'], 'Net AOV': row['contr_aov']}.items(), key=lambda x: x[1])[0], axis=1)
        return df

    shop_imp = compute_contribs(shop_pivot, 'shop')
    device_imp = compute_contribs(device_pivot, 'device_platform')
    mix = df.copy()
    mix['shop_device'] = mix['shop'] + ' | ' + mix['device_platform']
    mix_metrics = compute_bucket_metrics_by_level(mix, 'shop_device')
    mix_pivot = pivot_metrics(mix_metrics, 'shop_device').sort_values('total_visitors_Test', ascending=False)
    mix_imp = compute_contribs(mix_pivot, 'shop_device')

    insights = []
    segments = [
        ('Shop', shop_imp, 'shop'),
        ('Device', device_imp, 'device_platform'),
        ('Shop & Device', mix_imp, 'shop_device')
    ]
    for name, imp, col in segments:
        best = imp.nlargest(1, 'net_sales_impact')
        worst = imp.nsmallest(1, 'net_sales_impact')
        insights.append(
            f"**{name}**: Best segment “{best.iloc[0][col]}” with impact {best.iloc[0]['net_sales_impact']:.2f} (main contributor: {best.iloc[0]['main_contributor']}); "
            f"Worst segment “{worst.iloc[0][col]}” with impact {worst.iloc[0]['net_sales_impact']:.2f} (main contributor: {worst.iloc[0]['main_contributor']})."
        )

    st.markdown("**Segment Impact Insights:**")
    for bullet in insights:
        st.markdown(f"- {bullet}")

    with st.expander("📌 Segment Impact Analysis", expanded=False):
        st.subheader("Shop Segments")
        st.table(shop_imp.set_index('shop')[['net_sales_impact', 'contr_cr', 'contr_opc', 'contr_aov', 'main_contributor']])
        st.subheader("Device Segments")
        st.table(device_imp.set_index('device_platform')[['net_sales_impact', 'contr_cr', 'contr_opc', 'contr_aov', 'main_contributor']])
        st.subheader("Shop & Device Mix Segments")
        st.table(mix_imp.set_index('shop_device')[['net_sales_impact', 'contr_cr', 'contr_opc', 'contr_aov', 'main_contributor']])


if __name__ == "__main__":
    main()
