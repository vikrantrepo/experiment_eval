import pandas as pd
import numpy as np
import streamlit as st
import subprocess
import os
from pathlib import Path
from webbrowser import open as open_browser
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

# -------------------- AUTO CLEANUP --------------------
lock_path = Path(".streamlit_run.lock")
if lock_path.exists():
    lock_path.unlink()

# -------------------- DATA LOAD & CLEAN --------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'buckets', 'exposed_visitor_id', 'net_sales', 'order_id', 'order_status', 'device_platform', 'shop'}
    missing = required.difference(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df['net_sales'] = df['net_sales'].fillna(0.0)
    df['order_id'] = df['order_id'].fillna(0).astype(int)
    df['order_status'] = df['order_status'].fillna('Unknown').astype(str)
    return df

# -------------------- METRICS FUNCTIONS --------------------
def compute_bucket_metrics(grp: pd.core.groupby.DataFrameGroupBy) -> dict:
    total_visitors = grp['exposed_visitor_id'].nunique()
    converters = grp[(grp['order_id'] > 0) & grp['order_status'].isin(['L', 'O'])]['exposed_visitor_id'].nunique()
    orders_all = grp[grp['order_id'] > 0]['order_id'].nunique()
    orders_lo = grp[grp['order_status'].isin(['L', 'O'])]['order_id'].nunique()
    sales_sum = grp['net_sales'].sum()
    cancels = grp[grp['order_status'] == 'S']['order_id'].nunique()
    denom = orders_all if orders_all > 0 else None
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
        'total_net_sales': round(sales_sum, 2)
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
def bootstrap_rpev(df: pd.DataFrame, n_iters=1000, seed=42):
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

def conversion_z_test(df: pd.DataFrame):
    df['converted'] = df['order_id'] > 0
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    count = np.array([summary.loc['Test', 'sum'], summary.loc['Control', 'sum']])
    nobs = np.array([summary.loc['Test', 'count'], summary.loc['Control', 'count']])
    z, p = proportions_ztest(count, nobs)
    return z, p

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
    sorted_df = df.sort_values('total_visitors_Test', ascending=False)
    for col in cols:
        if col in sorted_df.columns:
            st.write(f"**{col.replace('_', ' ').title()}**")
            base = alt.Chart(sorted_df).encode(
                x=alt.X(index_col, sort=list(sorted_df[index_col])),
                y=alt.Y(col, title=col.replace('_', ' ').title()),
                tooltip=[index_col, col]
            )
            bars = base.mark_bar()
            fmt = ".0f" if col == 'conversion_rate_diff_bps' else (".1%" if col in ['net_aov_rel_diff', 'orders_per_converter_rel_diff'] else ".2f")
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                fontSize=12
            ).encode(
                text=alt.Text(col, format=fmt),
                color=alt.condition(alt.datum[col] < 0, alt.value("red"), alt.value("green"))
            )
            chart = (bars + text).properties(width=400, height=300)
            st.altair_chart(chart, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Experiment Dashboard", layout="wide")
    st.title("📊 Experiment Results")
    path = st.file_uploader("Upload CSV", type='csv')
    if not path:
        st.info("Please upload your checkout_shop_device.csv file.")
        return
    df = load_and_clean(path)

    # Filters (hidden in expander)
    with st.expander("🔍 Filter Options", expanded=False):
        shops = sorted(df['shop'].unique())
        devs = sorted(df['device_platform'].unique())
        sel_shops = st.multiselect("Shops", shops, default=shops)
        sel_devs = st.multiselect("Devices", devs, default=devs)
    df = df[df['shop'].isin(sel_shops) & df['device_platform'].isin(sel_devs)][df['shop'].isin(sel_shops) & df['device_platform'].isin(sel_devs)]

    # Grand totals
    st.subheader("🏁 Overall Metrics by Bucket")
    totals_df = get_bucket_totals(df)
    st.table(totals_df)

    # Statistical Tests Summary
    obs, p_boot, ci, diffs = bootstrap_rpev(df)
    z, p_z = conversion_z_test(df)
    (u_o, p_o), (u_a, p_a) = mann_whitney_tests(df)

    stats_summary = pd.DataFrame([
        {
            'Test': 'Revenue per Visitor (Bootstrap)',
            'Statistic': f"{obs:.4f}",
            'P-value': p_boot,
            'CI Lower': ci[0],
            'CI Upper': ci[1],
            'Significant': 'Yes' if p_boot < 0.05 else 'No'
        },
        {
            'Test': 'Conversion Rate (Z-test)',
            'Statistic': f"{z:.4f}",
            'P-value': p_z,
            'CI Lower': np.nan,
            'CI Upper': np.nan,
            'Significant': 'Yes' if p_z < 0.05 else 'No'
        },
        {
            'Test': 'Orders per Converter (Mann-Whitney)',
            'Statistic': f"{u_o:.2f}",
            'P-value': p_o,
            'CI Lower': np.nan,
            'CI Upper': np.nan,
            'Significant': 'Yes' if p_o < 0.05 else 'No'
        },
        {
            'Test': 'Net AOV (Mann-Whitney)',
            'Statistic': f"{u_a:.2f}",
            'P-value': p_a,
            'CI Lower': np.nan,
            'CI Upper': np.nan,
            'Significant': 'Yes' if p_a < 0.05 else 'No'
        }
    ])

    st.subheader("🔬 Statistical Tests Summary")
    st.table(stats_summary.set_index('Test'))

    # Visuals and detailed tests
    fig, ax = plt.subplots()
    ax.hist(diffs, bins=50, alpha=0.7)
    ax.axvline(obs, color='red', linestyle='--')
    ax.axvline(ci[0], color='gray', linestyle=':')
    ax.axvline(ci[1], color='gray', linestyle=':')
    st.pyplot(fig)

    # Level metrics
    shop_metrics = compute_bucket_metrics_by_level(df, 'shop')
    device_metrics = compute_bucket_metrics_by_level(df, 'device_platform')
    shop_pivot = pivot_metrics(shop_metrics, 'shop').sort_values('total_visitors_Test', ascending=False)
    device_pivot = pivot_metrics(device_metrics, 'device_platform').sort_values('total_visitors_Test', ascending=False)

    # Shop-Level Table
    st.subheader("🛒 Shop-Level Metrics")
    st.dataframe(
        shop_pivot.style.format({
            'conversion_rate_diff_bps': '{:.0f}',
            'net_aov_rel_diff': '{:.1%}',
            'orders_per_converter_rel_diff': '{:.1%}',
            'net_sales_per_visitor_abs_diff': '{:.2f}'
        }),
        use_container_width=True
    )

    # Device-Level Table
    st.subheader("📱 Device-Level Metrics")
    st.dataframe(
        device_pivot.style.format({
            'conversion_rate_diff_bps': '{:.0f}',
            'net_aov_rel_diff': '{:.1%}',
            'orders_per_converter_rel_diff': '{:.1%}',
            'net_sales_per_visitor_abs_diff': '{:.2f}'
        }),
        use_container_width=True
    )

    # Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Shop-Level Visuals")
        show_visuals(shop_pivot, 'shop')
    with col2:
        st.subheader("📊 Device-Level Visuals")
        show_visuals(device_pivot, 'device_platform')


    # -------------------- COMBINED SHOP+DEVICE SEGMENTS --------------------
    # Build combined segment identifier
    df['segment'] = df['shop'] + ' | ' + df['device_platform']
    comb_records = []
    for (seg, bucket), grp in df.groupby(['segment', 'buckets']):
        total_visitors = grp['exposed_visitor_id'].nunique()
        sales_sum = grp['net_sales'].sum()
        nspv = sales_sum / total_visitors if total_visitors else 0
        comb_records.append({'segment': seg, 'bucket': bucket,
                              'total_visitors': total_visitors, 'nspv': nspv})
    comb_df = pd.DataFrame(comb_records)
    pivot_comb = comb_df.pivot(index='segment', columns='bucket', values=['total_visitors', 'nspv'])
    pivot_comb.columns = [f"{metric}_{bucket}" for metric, bucket in pivot_comb.columns]
    pivot_comb = pivot_comb.reset_index()
    pivot_comb['nspv_diff'] = pivot_comb['nspv_Test'] - pivot_comb['nspv_Control']
    pivot_comb['impact'] = pivot_comb['nspv_diff'] * pivot_comb['total_visitors_Test']
    # Top 3 positive & negative
    top3_comb = pivot_comb.nlargest(3, 'impact')
    worst3_comb = pivot_comb.nsmallest(3, 'impact')

    st.subheader("🔀 Combined Shop+Device Segments Impact")
    st.markdown("**Top 3 Positive Segments**")
    st.table(top3_comb[['segment', 'total_visitors_Test', 'nspv_Control', 'nspv_Test', 'nspv_diff', 'impact']])
    st.markdown("**Top 3 Negative Segments**")
    st.table(worst3_comb[['segment', 'total_visitors_Test', 'nspv_Control', 'nspv_Test', 'nspv_diff', 'impact']])
    # -------------------- SEGMENT IMPACT & CONTRIBUTIONS --------------------
    # Compute contributions for shops, devices, and combined segments
    def compute_segment_contrib(data, level):
        recs = []
        for (seg, bucket), grp in data.groupby([level, 'buckets']):
            visitors = grp['exposed_visitor_id'].nunique()
            converters = grp[(grp['order_id']>0) & grp['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique()
            orders = grp[grp['order_status'].isin(['L','O'])]['order_id'].nunique()
            sales = grp['net_sales'].sum()
            conv_rate = converters / visitors if visitors>0 else 0
            aov = sales / orders if orders>0 else 0
            ord_per_conv = orders / converters if converters>0 else 0
            nspv = sales / visitors if visitors>0 else 0
            recs.append({
                level: seg,
                'bucket': bucket,
                'visitors': visitors,
                'conv_rate': conv_rate,
                'aov': aov,
                'ord_per_conv': ord_per_conv,
                'nspv': nspv
            })
        dfm = pd.DataFrame(recs)
        pivot = dfm.pivot(index=level, columns='bucket')
        pivot.columns = ['_'.join(col) for col in pivot.columns]
        pivot = pivot.reset_index()
        pivot['diff_conv'] = pivot['conv_rate_Test'] - pivot['conv_rate_Control']
        pivot['diff_aov'] = pivot['aov_Test'] - pivot['aov_Control']
        pivot['diff_ord'] = pivot['ord_per_conv_Test'] - pivot['ord_per_conv_Control']
        pivot['conv_contrib'] = pivot['diff_conv'] * pivot['aov_Control'] * pivot['ord_per_conv_Control'] * pivot['visitors_Test']
        pivot['aov_contrib'] = pivot['diff_aov'] * pivot['conv_rate_Control'] * pivot['ord_per_conv_Control'] * pivot['visitors_Test']
        pivot['ord_contrib'] = pivot['diff_ord'] * pivot['conv_rate_Control'] * pivot['aov_Control'] * pivot['visitors_Test']
        pivot['impact'] = pivot['nspv_Test'] - pivot['nspv_Control']
        pivot['total_impact'] = pivot['impact'] * pivot['visitors_Test']
        pivot['main_contributor'] = pivot[['conv_contrib','aov_contrib','ord_contrib']].abs().idxmax(axis=1).map({
            'conv_contrib': 'Conversion',
            'aov_contrib': 'AOV',
            'ord_contrib': 'Orders per converted visitor'
        })
        return pivot

    shop_df = compute_segment_contrib(df, 'shop')
    device_df = compute_segment_contrib(df, 'device_platform')
    df['segment'] = df['shop'] + ' | ' + df['device_platform']
    combined_df = compute_segment_contrib(df, 'segment')

    # Get top and worst 3
    shop_top = shop_df.nlargest(3, 'total_impact')
    shop_worst = shop_df.nsmallest(3, 'total_impact')
    device_top = device_df.nlargest(3, 'total_impact')
    device_worst = device_df.nsmallest(3, 'total_impact')
    combined_top = combined_df.nlargest(3, 'total_impact')
    combined_worst = combined_df.nsmallest(3, 'total_impact')

    # Display tables
    st.subheader("🏆 Segment Impact & Contribution Tables")
    st.markdown("**Top 3 Shops**")
    st.dataframe(shop_top[['shop','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)
    st.markdown("**Worst 3 Shops**")
    st.dataframe(shop_worst[['shop','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)
    st.markdown("**Top 3 Devices**")
    st.dataframe(device_top[['device_platform','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)
    st.markdown("**Worst 3 Devices**")
    st.dataframe(device_worst[['device_platform','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)
    st.markdown("**Top 3 Combined Segments**")
    st.dataframe(combined_top[['segment','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)
    st.markdown("**Worst 3 Combined Segments**")
    st.dataframe(combined_worst[['segment','visitors_Control','visitors_Test','nspv_Control','nspv_Test','impact','total_impact','conv_contrib','aov_contrib','ord_contrib','main_contributor']], use_container_width=True)

        # Insights summary
    st.subheader("💡 Insights Summary")
    insights = []
    # Shop insights
    insights.append(f"Top shop '{shop_top.iloc[0]['shop']}' added {shop_top.iloc[0]['total_impact']:.0f} in net sales, driven mainly by {shop_top.iloc[0]['main_contributor']}." )
    insights.append(f"Worst shop '{shop_worst.iloc[0]['shop']}' lost {abs(shop_worst.iloc[0]['total_impact']):.0f} in net sales, impacted mainly by {shop_worst.iloc[0]['main_contributor']}.")
    # Device insights
    insights.append(f"Top device '{device_top.iloc[0]['device_platform']}' added {device_top.iloc[0]['total_impact']:.0f}, led by {device_top.iloc[0]['main_contributor']}.")
    insights.append(f"Worst device '{device_worst.iloc[0]['device_platform']}' changed by {device_worst.iloc[0]['total_impact']:.0f}, driven by {device_worst.iloc[0]['main_contributor']}.")
    # Combined insights
    insights.append(f"Segment '{combined_top.iloc[0]['segment']}' saw the highest lift of {combined_top.iloc[0]['total_impact']:.0f}, thanks mostly to {combined_top.iloc[0]['main_contributor']}.")
    insights.append(f"Segment '{combined_worst.iloc[0]['segment']}' experienced the largest drop of {abs(combined_worst.iloc[0]['total_impact']):.0f}, due to {combined_worst.iloc[0]['main_contributor']}.")
    # Global dynamic insight
    ctrl = df[df['buckets']=='Control']
    test = df[df['buckets']=='Test']
    v_ctrl, v_test = ctrl['exposed_visitor_id'].nunique(), test['exposed_visitor_id'].nunique()
    ns_ctrl, ns_test = ctrl['net_sales'].sum(), test['net_sales'].sum()
    overall_diff = (ns_test/v_test if v_test else 0) - (ns_ctrl/v_ctrl if v_ctrl else 0)
    # compute global contributions
    conv_ctrl = ctrl[(ctrl['order_id']>0)&ctrl['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique()/v_ctrl if v_ctrl else 0
    conv_test = test[(test['order_id']>0)&test['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique()/v_test if v_test else 0
    aov_ctrl = ns_ctrl/(ctrl[ctrl['order_status'].isin(['L','O'])]['order_id'].nunique()) if ctrl[ctrl['order_status'].isin(['L','O'])]['order_id'].nunique() else 0
    aov_test = ns_test/(test[test['order_status'].isin(['L','O'])]['order_id'].nunique()) if test[test['order_status'].isin(['L','O'])]['order_id'].nunique() else 0
    ord_ctrl = (ctrl[ctrl['order_status'].isin(['L','O'])]['order_id'].nunique()/ctrl[(ctrl['order_id']>0)&ctrl['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique()) if ctrl[(ctrl['order_id']>0)&ctrl['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique() else 0
    ord_test = (test[test['order_status'].isin(['L','O'])]['order_id'].nunique()/test[(test['order_id']>0)&test['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique()) if test[(test['order_id']>0)&test['order_status'].isin(['L','O'])]['exposed_visitor_id'].nunique() else 0
    diff_conv, diff_aov, diff_ord = conv_test-conv_ctrl, aov_test-aov_ctrl, ord_test-ord_ctrl
    contrib_conv = diff_conv * aov_ctrl * ord_ctrl * v_test
    contrib_aov = diff_aov * conv_ctrl * ord_ctrl * v_test
    contrib_ord = diff_ord * conv_ctrl * aov_ctrl * v_test
    contribs = {'Conversion': abs(contrib_conv), 'AOV': abs(contrib_aov), 'Orders per converted visitor': abs(contrib_ord)}
    main_global = max(contribs, key=contribs.get)
    insights.append(f"Overall, net sales per visitor changed by {overall_diff:.2f}. Contributions: Conversion={contrib_conv:.2f}, AOV={contrib_aov:.2f}, Orders={contrib_ord:.2f}. Main driver: {main_global}.")
    for item in insights:
        st.markdown(f"- {item}")(f"- {item}"):
        st.markdown(f"- {item}")

if __name__ == "__main__":
    main()
    if os.getenv("STREAMLIT_SERVER_PORT") or os.getenv("RUN_FROM_STREAMLIT") == "1":
        main()
    else:
        lock_path.write_text("running")
        env = os.environ.copy()
        env["RUN_FROM_STREAMLIT"] = "1"
        subprocess.Popen(["streamlit","run",os.path.abspath(__file__)], env=env)
