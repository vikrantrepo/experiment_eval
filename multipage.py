import streamlit as st
import datetime
import pandas as pdo
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

st.set_page_config(page_title="SQL Builder & Experiment Dashboard", layout="wide")
st.title("🛠️ SQL Builder & 📊 Experiment Dashboard")

tab1, tab2, tab3 = st.tabs(["SQL Query Builder", "Experiment Dashboard", "Documentation"])

# --------------- TAB 1: SQL QUERY BUILDER ---------------
with tab1:
    st.header("SQL Query Builder")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Reference Start Date", value=datetime.date(2025, 5, 5))
        date_time = st.text_input("Test Start Timestamp", value="2025-05-05 11:00:00")
        control_mvvar3 = st.text_input("Control mvvar3", value="%4747_new_login_registration_page^va:tru%")
        post_evar59 = st.multiselect("Shops (post_evar59)", options=[
            'zooplus.de', 'zooplus.pl', 'zooplus.fr', 'zooplus.it', 'zooplus.nl', 'zooplus.es',
            'zooplus.co.uk', 'zooplus.hu', 'zooplus.ro', 'zoohit.cz', 'zooplus.se', 'zooplus.be',
            'zooplus.ch', 'bitiba.de', 'zooplus.pt', 'zooplus.dk', 'zooplus.at', 'bitiba.pl',
            'zoohit.sk', 'zooplus.fi', 'bitiba.fr', 'bitiba.it', 'zooplus.no', 'bitiba.cz',
            'bitiba.es', 'bitiba.nl', 'zooplus.hr', 'zooplus.bg', 'zooplus.ie', 'zooplus.gr',
            'zoohit.si', 'zooplus.com', 'bitiba.co.uk', 'bitiba.se', 'bitiba.ch', 'bitiba.dk',
            'zoochic-eu.ru', 'bitiba.fi', 'bitiba.be', 'bitiba.com'
        ], default=['zooplus.de'])

    with col2:
        end_date = st.date_input("Reference End Date", value=datetime.date(2025, 5, 14))
        test_mvvar3 = st.text_input("Test mvvar3", value="%4747_new_login_registration_page^va:fal%")
        post_evar42 = st.multiselect("Devices (post_evar42)", options=["notApp", "ios", "android"], default=["notApp"])
        post_evar58 = st.text_input("URL Paths (comma-separated) *Optional*", value="/checkout/login.htm, /checkout/login, /checkout/register")
        post_evar22 = st.text_input("Page Types (comma-separated) *Optional*", value="checkout")

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
        a.post_evar59 as shop,
	    case when a.post_evar42='notApp' and mobile_id=0 then 'Web_desktop'
	         else case when a.post_evar42='notApp' and mobile_id>0 then 'Web_mobile'
	             else case when a.post_evar42='android' then 'App_android'
	                 else case when  a.post_evar42='ios' then 'App_iOS' else 'Undefined' 
        end end end end as device_platform, 
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
  GROUP BY 1, 2, 3, 4
  UNION ALL
  SELECT 'Test' AS bucket,
  	b.post_evar59 as shop,
        case when b.post_evar42='notApp' and mobile_id=0 then 'Web_desktop'
	         else case when b.post_evar42='notApp' and mobile_id>0 then 'Web_mobile'
	             else case when b.post_evar42='android' then 'App_android'
	                 else case when  b.post_evar42='ios' then 'App_iOS' else 'Undefined' 
        end end end end as device_platform, 
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
  GROUP BY 1, 2, 3, 4
),
multi_bucket_visitors AS (
	  SELECT visitor_id FROM raw_exposures GROUP BY visitor_id HAVING COUNT(DISTINCT bucket) = 1
),
bucketed_visitors_first_exposure AS (
	  SELECT re.shop, re.bucket AS buckets, re.device_platform, re.visitor_id AS exposed_visitor_id, first_exposure_timestamp
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
      AND so.so_order_dt BETWEEN exp.first_exposure_timestamp AND timestamp '{ref_end} 23:59:59'
  GROUP BY 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
),
conversion_summary AS (
  SELECT attributed_visitor_id AS converted_visitor_id,
        order_id, net_sales, order_status, order_counter, cm1, cm2, nc_order_f, tnc_order_f
  FROM orders_deduped
)
--select distinct shop, buckets, device_platform, DENSE_RANK() OVER (ORDER BY exposed_visitor_id) AS exposed_visitor_id, DENSE_RANK() OVER (ORDER BY coalesce(order_id,0)) AS order_id,coalesce(net_sales,0) net_sales, coalesce(cs.cm1,0) as cm1,coalesce(cs.cm2,0)  as cm2,cs.order_status from bucketed_visitors_first_exposure LEFT JOIN conversion_summary cs ON exposed_visitor_id = cs.converted_visitor_id
SELECT exp.buckets, --exp.shop,exp.buckets, exp.device_platform,
       COUNT(DISTINCT exp.exposed_visitor_id) AS exposed_visitors,
       COUNT(DISTINCT CASE WHEN cs.order_status IN ('L','O') THEN cs.converted_visitor_id ELSE NULL end ) AS converted_visitors_L_O_post_exposure,
       COUNT(DISTINCT CASE WHEN cs.order_status IN ('L') THEN cs.converted_visitor_id ELSE NULL end ) AS converted_visitors_L_post_exposure,
       ROUND(COUNT(DISTINCT CASE WHEN cs.order_status IN ('L','O') THEN cs.converted_visitor_id ELSE NULL end) * 1.0 / COUNT(DISTINCT exp.exposed_visitor_id), 4) AS converted_visitor_share_L_O,
       ROUND(COUNT(DISTINCT CASE WHEN cs.order_status IN ('L') THEN cs.converted_visitor_id ELSE NULL end) * 1.0 / COUNT(DISTINCT exp.exposed_visitor_id), 4) AS converted_visitor_share_L,
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
GROUP BY 1--,2,3
ORDER BY 1
""".strip()
        return query

    st.subheader("Generated SQL Query")
    # build the SQL
    sql_code = build_sql()
    clean_sql = "\n".join(line for line in sql_code.splitlines() if line.strip())
    st.code(clean_sql, language="sql")


# --------------- TAB 2: EXPERIMENT DASHBOARD ---------------
with tab2:
    st.header("Experiment Eval")
    import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm  # Added for CI calculation

# -------------------- DATA LOAD & CLEAN --------------------
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

# -------------------- METRICS FUNCTIONS --------------------
def compute_bucket_metrics(grp: pd.core.groupby.DataFrameGroupBy) -> dict:
    total_visitors = grp['exposed_visitor_id'].nunique()
    converters_L = grp[(grp['order_id'] > 1) & grp['order_status'].isin(['L'])]['exposed_visitor_id'].nunique()
    orders_all = grp[grp['order_id'] > 1]['order_id'].nunique()
    orders_l = grp[grp['order_status'].isin(['L'])]['order_id'].nunique()
    sales_sum = grp['net_sales'].sum()
    cancels = grp[grp['order_status'] == 'S']['order_id'].nunique()
    denom = orders_all if orders_all > 0 else None

    # New: cm1 and cm2 percentages
    sum_cm1 = grp['cm1'].sum()
    sum_cm2 = grp['cm2'].sum()
    cm1_per_vis = sum_cm1 / total_visitors if total_visitors else 0
    cm2_per_vis = sum_cm2 / total_visitors if total_visitors else 0
    cm1_per_sales = sum_cm1 / sales_sum if sales_sum else 0
    cm2_per_sales = sum_cm2 / sales_sum if sales_sum else 0

    return {
        'total_visitors': total_visitors,
        'converting_visitors': converters_L,
        'conversion_rate': round(converters_L/total_visitors, 4) if total_visitors else 0,
        'orders_all': orders_all,
        'orders_L': orders_l,
        'net_aov': round(sales_sum/orders_l, 4) if orders_l else 0,
        'orders_per_converting_visitor': round(orders_l/converters_L, 4) if converters_L else 0,
        'share_of_cancelled_orders': round(cancels/denom, 4) if denom else 0,
        'net_sales_per_visitor': round(sales_sum/total_visitors, 4) if total_visitors else 0,
        'total_net_sales': round(sales_sum, 2),
        # new metrics
        'cm1_per_total_visitors': cm1_per_vis,
        'cm2_per_total_visitors': cm2_per_vis,
        'cm1_per_total_net_sales': cm1_per_sales,
        'cm2_per_total_net_sales': cm2_per_sales
    }

# Aggregated bucket metrics

def get_bucket_totals(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for bucket, grp in df.groupby('buckets'):
        rec = compute_bucket_metrics(grp)
        rec['bucket'] = bucket
        records.append(rec)
    totals = pd.DataFrame(records).set_index('bucket')
    ordered = ['Control', 'Test']
    return totals.reindex(ordered)

# Metrics by level (shop or device)

def compute_bucket_metrics_by_level(df, level):
    records = []
    for (lvl_val, bucket), grp in df.groupby([level, 'buckets']):
        rec = compute_bucket_metrics(grp)
        rec[level] = lvl_val
        rec['buckets'] = bucket
        records.append(rec)
    return pd.DataFrame(records).sort_values([level, 'buckets'])

# Pivot & differences

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

# -------------------- STATISTICAL TESTS --------------------
def bootstrap_rpev(df: pd.DataFrame, n_iters=12000):
    visitor_sales = df.groupby(['buckets', 'exposed_visitor_id'], as_index=False)['net_sales'].sum()
    test = visitor_sales.loc[visitor_sales.buckets == 'Test', 'net_sales'].values
    ctrl = visitor_sales.loc[visitor_sales.buckets == 'Control', 'net_sales'].values
    obs = test.mean() - ctrl.mean()
    rng = np.random.default_rng()
    diffs = np.array([
        rng.choice(test, size=len(test), replace=True).mean() -
        rng.choice(ctrl, size=len(ctrl), replace=True).mean()
        for _ in range(n_iters)
    ])
    p_val = np.mean(np.abs(diffs) >= abs(obs))
    ci = np.percentile(diffs, [2.5, 97.5])
    return obs, p_val, ci, diffs


def conversion_z_test(df: pd.DataFrame, alpha=0.05):
    df['converted'] = df['order_id'] > 1
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

# -------------------- VISUALIZATION HELPERS --------------------
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
                    axis=alt.Axis(labelAngle=-90, labelAlign='right', labelLimit=200)
                ),
                y=alt.Y(
                    col,
                    title=col.replace('_', ' ').title(),
                    axis=alt.Axis(labelAngle=0, labelAlign='right', titlePadding=10)
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

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Experiment Dashboard", layout="wide")
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

    # -------------------- OUTLIER REMOVAL --------------------
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

    outlier_buckets = df_lo_overall[df_lo_overall['exposed_visitor_id'].isin(outlier_ids)][['exposed_visitor_id','buckets']].drop_duplicates()
    excluded_counts = outlier_buckets.groupby('buckets')['exposed_visitor_id'].nunique().reindex(['Control','Test'], fill_value=0)
    st.write(f"**Excluded Visitors:** Control: {excluded_counts.loc['Control']}, Test: {excluded_counts.loc['Test']} (all visitors above 99.9th percentile in AOV or orders per converter)")

    # -------------------- OVERALL METRICS --------------------
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
        'orders_L': '{:,.0f}',
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
    #st.dataframe(styled, use_container_width=True)
    st.table(styled)

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
    total_sales_test = totals_df.loc['Test',    'total_net_sales'] 
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
    stats_summary['Impact'] = [
    f"€{int(net_sales_impact)}",
    f"€{int(contr_cr)}",
    f"€{int(contr_opc)}",
    f"€{int(contr_aov)}"]

    # ─── BAYESIAN ANALYSIS ──────────────────────────────────────────────────

    def bayesian_bootstrap_diff(ctrl_vals, test_vals, n_iters=12000, cred_mass=0.95):
        rng = np.random.default_rng()
        diffs = []
        for _ in range(n_iters):
            w_c = rng.dirichlet(np.ones(len(ctrl_vals)))
            w_t = rng.dirichlet(np.ones(len(test_vals)))
            diffs.append((test_vals * w_t).sum() - (ctrl_vals * w_c).sum())
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [(1-cred_mass)/2*100, (1+cred_mass)/2*100])
        prob = (diffs > 0).mean()
        return prob, lo, hi

    metrics = {
        'Revenue per Visitor':          'net_sales_per_visitor',
        'CM1 per Visitor':              'cm1_per_total_visitors',
        'CM2 per Visitor':              'cm2_per_total_visitors',
        'CM1 Share of Net Sales':       'cm1_per_total_net_sales',
        'CM2 Share of Net Sales':       'cm2_per_total_net_sales',
    }

    rows = []
    # totals_df has exactly those columns
    for name, col in metrics.items():
        # point estimates
        ctrl = totals_df.loc['Control', col]
        test = totals_df.loc['Test', col]

        # for bootstrap we need the raw per-visitor series:
        #  - revenue: group df.net_sales by exposed_visitor_id
        #  - cm1/cm2: group df.cm1 / df.cm2 by exposed_visitor_id
        if col == 'net_sales_per_visitor':
            ctrl_series = (
                df[df.buckets=='Control']
                  .groupby('exposed_visitor_id')['net_sales']
                  .sum().values
            )
            test_series = (
                df[df.buckets=='Test']
                  .groupby('exposed_visitor_id')['net_sales']
                  .sum().values
            )
        elif col == 'cm1_per_total_visitors':
            ctrl_series = df[df.buckets=='Control'].groupby('exposed_visitor_id')['cm1'].sum().values
            test_series = df[df.buckets=='Test'].groupby('exposed_visitor_id')['cm1'].sum().values
        elif col == 'cm2_per_total_visitors':
            ctrl_series = df[df.buckets=='Control'].groupby('exposed_visitor_id')['cm2'].sum().values
            test_series = df[df.buckets=='Test'].groupby('exposed_visitor_id')['cm2'].sum().values
        else:
            # for share metrics you can bootstrap the same series and transform
            # or just approximate with point masses:
            ctrl_series = np.full(1000, ctrl)
            test_series = np.full(1000, test)

        p, lo, hi = bayesian_bootstrap_diff(ctrl_series, test_series)
        # compute impact (integer)
        diff = test - ctrl
        if name == 'Revenue per Visitor':
            impact = int(diff * total_vis_test)
        elif name in ('CM1 per Visitor', 'CM2 per Visitor'):
            impact = int(diff * total_vis_test)
        else:  # CM1/CM2 share of net sales
            impact = int(diff * total_sales_test)

        rows.append({
            'Metric': name,
            'P(Test > Control)':      f"{p*100:.2f}%",
            'CI Lower':               f"{lo:.4f}",
            'CI Upper':               f"{hi:.4f}",
            'Impact':                 f"€{impact}"
        })

    bayes_summary = pd.DataFrame(rows).set_index('Metric')

    # --- Build and display Insight Summary ---
    # --- Frequentist Insight ---
    ctrl_nspv = totals_df.loc['Control','net_sales_per_visitor']
    test_nspv = totals_df.loc['Test','net_sales_per_visitor']
    delta_nspv = test_nspv - ctrl_nspv
    rel_nspv = (delta_nspv / ctrl_nspv) * 100 if ctrl_nspv != 0 else np.nan
    sig_nrpv = "significant" if p_boot < 0.05 else "not significant"

    ctrl_cr = totals_df.loc['Control','conversion_rate']
    test_cr = totals_df.loc['Test','conversion_rate']
    delta_cr = test_cr - ctrl_cr
    rel_cr = (delta_cr / ctrl_cr) * 100 if ctrl_cr != 0 else np.nan
    sig_cr = "significant" if p_z < 0.05 else "not significant"

    ctrl_opc = totals_df.loc['Control','orders_per_converting_visitor']
    test_opc = totals_df.loc['Test','orders_per_converting_visitor']
    delta_opc = test_opc - ctrl_opc
    rel_opc = (delta_opc / ctrl_opc) * 100 if ctrl_opc != 0 else np.nan
    sig_opc = "significant" if p_o < 0.05 else "not significant"

    ctrl_aov = totals_df.loc['Control','net_aov']
    test_aov = totals_df.loc['Test','net_aov']
    delta_aov = test_aov - ctrl_aov
    rel_aov = (delta_aov / ctrl_aov) * 100 if ctrl_aov != 0 else np.nan
    sig_aov = "significant" if p_a < 0.05 else "not significant"

    insight_frequentist = (
        f"**Net revenue per visitor** changed by **€{delta_nspv:.2f} / {rel_nspv:+.2f}%** "
        f"(C: €{ctrl_nspv:.2f}, T: €{test_nspv:.2f}) "
        f"(p={p_boot:.3f}, {sig_nrpv}). "
        f"NRPV components: "
        f"**Conversion rate** changed by **{delta_cr:.2%} / {rel_cr:+.2f}%** "
        f"(C: {ctrl_cr:.2%}, T: {test_cr:.2%}) (p={p_z:.3f}, {sig_cr}), "
        f"**Orders per converter** changed by **{delta_opc:.4f} / {rel_opc:+.2f}%** "
        f"(C: {ctrl_opc:.4f}, T: {test_opc:.4f}) (p={p_o:.3f}, {sig_opc}), "
        f"**Net AOV** changed by **€{delta_aov:.2f} / {rel_aov:+.2f}%** "
        f"(C: €{ctrl_aov:.2f}, T: €{test_aov:.2f}) (p={p_a:.3f}, {sig_aov})."
    )

    # --- Bayesian Insight ---
    bayesian_metrics = []
    for metric_name, col in metrics.items():
        ctrl = totals_df.loc['Control', col]
        test = totals_df.loc['Test', col]
        diff = test - ctrl
        rel = (diff / ctrl) * 100 if ctrl != 0 else np.nan
        prob = bayes_summary.loc[metric_name, 'P(Test > Control)']
        impact = bayes_summary.loc[metric_name, 'Impact']
        # Value formatting based on metric type
        if 'Share' in metric_name:
            # shares in percent
            val_fmt = lambda x: f"{x:.2%}"
            abs_fmt = lambda x: f"{x:+.2%}"
        elif 'Conversion' in metric_name:
            val_fmt = lambda x: f"{x:.2%}"
            abs_fmt = lambda x: f"{x:+.2%}"
        else:
            val_fmt = lambda x: f"€{x:.4f}" if abs(x) < 100 else f"€{x:,.0f}"
            abs_fmt = lambda x: f"€{x:+.4f}" if abs(x) < 100 else f"€{x:+,.0f}"
        bayesian_metrics.append(
            f"**{metric_name}** changed by **{abs_fmt(diff)} / {rel:+.2f}%** (T: {val_fmt(test)}, C: {val_fmt(ctrl)}), "
            f"probability T > C: {prob}, impact: {impact}"
        )

    insight_bayesian = " | ".join(bayesian_metrics)

    # --- Display both insights ---
    st.markdown("### 🔎 Insight Summary")
    st.write(insight_frequentist)
    st.write(insight_bayesian)


    ctrl_nspv = totals_df.loc['Control', 'net_sales_per_visitor']
    test_nspv = totals_df.loc['Test',    'net_sales_per_visitor']
    delta_nspv = test_nspv - ctrl_nspv
    if delta_nspv < 0:
        loss = -delta_nspv

        # 2) required full‐offset CMx
        ctrl_cm1 = totals_df.loc['Control', 'cm1_per_total_visitors']
        ctrl_cm2 = totals_df.loc['Control', 'cm2_per_total_visitors']
        req_cm1 = ctrl_cm1 + loss
        req_cm2 = ctrl_cm2 + loss

        # 3) actual Test CMx
        test_cm1 = totals_df.loc['Test', 'cm1_per_total_visitors']
        test_cm2 = totals_df.loc['Test', 'cm2_per_total_visitors']
        gap_cm1 = test_cm1 - req_cm1
        gap_cm2 = test_cm2 - req_cm2

        st.markdown("**💡 Full‐offset breakeven CM1 & CM2 per Visitor**")
        st.write(
            f"- To fully offset the €{loss:.4f} RPV loss, you’d need CM1 ≥ **€{req_cm1:.4f}**; "
            f"actual: **€{test_cm1:.4f}** (Δ {gap_cm1:+.4f})\n"
            f"- To fully offset the €{loss:.4f} RPV loss, you’d need CM2 ≥ **€{req_cm2:.4f}**; "
            f"actual: **€{test_cm2:.4f}** (Δ {gap_cm2:+.4f})"
        )



    # render side‑by‑side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔬 Frequentist Tests : NS/ exposed visitor and respective components")
        st.table(stats_summary.set_index('Test'))
    with col2:
        st.subheader("🔭 Bayesian Analysis : Probability on Net sales and Margins")
        st.table(bayes_summary)

    # ────────────────────────────────────────────────────────────────────────
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

        # draw the boxplot
        visitor_stats.boxplot(column='net_aov', by='buckets', ax=ax2)
        ax2.set_title('Net AOV by Bucket')
        ax2.set_xlabel('')
        ax2.set_ylabel('Net AOV')
        plt.suptitle('')  # remove the automatic “by buckets” suptitle

        # compute quartiles per bucket
        quartiles = (
            visitor_stats
            .groupby('buckets')['net_aov']
            .quantile([0.25, 0.5, 0.75])
            .unstack(level=-1)    # columns will be [0.25, 0.5, 0.75]
        )

        # annotate each bucket: x positions are 1, 2, … by default
        for i, bucket in enumerate(quartiles.index, start=1):
            q1 = quartiles.loc[bucket, 0.25]
            med = quartiles.loc[bucket, 0.5]
            q3 = quartiles.loc[bucket, 0.75]

            # shift text slightly to the right so it doesn’t overlap the box
            x_text = i + 0.1

            ax2.text(x_text, q1, f"Q1: {q1:.2f}", va='center', fontsize=6)
            ax2.text(x_text, med, f"Med: {med:.2f}", va='center', fontsize=6)
            ax2.text(x_text, q3, f"Q3: {q3:.2f}", va='center', fontsize=6)

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

with tab3:
    st.header("SQL Builder & Experiment Dashboard")
    st.markdown("""
This Streamlit app consists of two main modules:

- **SQL Query Builder**  
- **Experiment Dashboard**

Each module lives in its own tab, and together they enable you to:
1. Quickly generate a parameterized SQL query over Adobe and sales data  
2. Upload the resulting query output and perform a full A/B-test evaluation, complete with metrics, statistical tests, and segment-level insights

---

### 1. SQL Query Builder

**Purpose:**  
Interactively configure experiment parameters and automatically generate the SQL you need to pull “exposure → conversion → revenue” data for Control vs Test groups.

**Inputs**

| Field                          | What it controls                                  |
| ------------------------------ | -------------------------------------------------- |
| Reference Start/End Date       | Date range over which to source raw exposures     |
| Test Start Timestamp           | Only include exposures after this exact timestamp |
| Control / Test `mvvar3`        | Regex pattern identifying Control vs Test buckets |
| Shops (`post_evar59`)          | Which zooplus domains to include                  |
| Devices (`post_evar42`)        | Filter by Web/iOS/Android                         |
| URL Paths (`post_evar58`)      | Optional comma-separated page URL filters         |
| Page Types (`post_evar22`)     | Optional comma-separated page-type filters        |

**Workflow**

1. User picks all parameters in the left and right columns.  
2. “Build SQL” function:
   - Assembles CTEs for:
     - Raw exposures (Control/Test buckets)
     - Filtering single-bucket visitors
     - Mapping visitors to customers
     - Deduplicating orders
     - Summarizing conversions
   - Joins exposures to conversions → metrics by bucket  
3. Generated SQL appears in a syntax-highlighted code block.  
4. Copy & paste this SQL into your data warehouse to export a CSV.

---

### 2. Experiment Dashboard

**Purpose:**  
Load the CSV from your SQL query, clean it, compute core metrics and statistical tests, visualize differences, and generate segment-level insights.

**A. Data Load & Cleaning**  
- Upload CSV via the file uploader.  
- Verifies required columns  
  `buckets, exposed_visitor_id, net_sales, order_id, order_status, device_platform, shop, cm1, cm2`  
- Fills missing values and ensures correct dtypes.

**B. Metric Computation**  
- `compute_bucket_metrics` returns:
  - Visitors, converting visitors, conversion rate  
  - Total orders, L/O orders, net AOV, orders per converter  
  - Cancellation rate  
  - Net sales per visitor & total net sales  
  - CM1/CM2 metrics (per visitor & per sales)  
- Helpers:
  - `get_bucket_totals(df)` → overall Control vs Test  
  - `compute_bucket_metrics_by_level(df, level)` → shop/device breakdown  
  - `pivot_metrics(...)` → side-by-side with deltas (bps, %)

**C. Outlier Removal**  
- Identifies top 0.1% of visitors by AOV or orders per converter  
- Excludes them and reports counts by bucket

**D. Statistical Tests**  
1. **Bootstrap RPV**: resamples per-visitor net sales → p-value & 95% CI  
2. **Conversion Z-test**: two-sample z-test on conversion flags  
3. **Mann-Whitney**: orders per converter & net AOV  
- Summarized in a table with statistic, p-value, CI, and significance flag

**E. Insights & Contribution Analysis**  
- **Overall Insight**:  
  - If RPV significant → reports sign and main driver (CR, OPC, or AOV)  
  - Otherwise → one-sentence summary with p-values for RPV, CR, AOV  
- **Revenue Impact Decomposition**:  
  - Calculates net-sales impact from each metric lever  
  - Identifies the primary driver

**F. Visualizations**  
- **Distribution & Boxplots**  
  - Bootstrap distribution of RPV deltas  
  - Boxplots: Net AOV & orders per converter by bucket  
- **Charts by Segment**  
  - Altair bar charts showing deltas (bps, rel/abs diffs) for shops and devices

**G. Segment-Level Tables & Insights**  
- **Tables**: shop-level and device-level pivot tables  
- **Segment Impact**: for each shop, device, and shop×device slice:
  - Net-sales impact  
  - Contribution breakdown (CR, OPC, AOV)  
  - Main contributor label  
- **Bullet-list Insights**: “Best” & “Worst” segments by impact
    """, unsafe_allow_html=True)
