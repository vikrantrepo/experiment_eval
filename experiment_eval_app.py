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

def conversion_z_test(df: pd.DataFrame, alpha=0.05):
    df['converted'] = df['order_id'] > 0
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    successes = np.array([summary.loc['Test', 'sum'], summary.loc['Control', 'sum']])
    nobs = np.array([summary.loc['Test', 'count'], summary.loc['Control', 'count']])
    p1 = successes[0] / nobs[0]
    p2 = successes[1] / nobs[1]
    diff = p1 - p2
    p_pool = successes.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1]))
    z = diff / se
    _, p = proportions_ztest(successes, nobs)
    z_alpha = norm.ppf(1 - alpha/2)
    ci = (diff - z_alpha * se, diff + z_alpha * se)
    return z, p, ci

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
                x=alt.X(
                    index_col,
                    sort=list(sorted_df[index_col]),
                    axis=alt.Axis(labelAngle=-45, labelAlign='right', labelLimit=200)
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

    # Overall Metrics
    st.subheader("🏁 Overall Metrics by Bucket")
    totals_df = get_bucket_totals(df)
    # Add absolute difference row for key metrics
    diff = pd.Series(index=totals_df.columns, name='Absolute Difference')
    diff['conversion_rate'] = round((totals_df.loc['Test','conversion_rate'] - totals_df.loc['Control','conversion_rate']) * 10000, 0)
    diff['net_aov'] = round(totals_df.loc['Test','net_aov'] - totals_df.loc['Control','net_aov'], 4)
    diff['orders_per_converting_visitor'] = round(totals_df.loc['Test','orders_per_converting_visitor'] - totals_df.loc['Control','orders_per_converting_visitor'], 4)
    diff['net_sales_per_visitor'] = round(totals_df.loc['Test','net_sales_per_visitor'] - totals_df.loc['Control','net_sales_per_visitor'], 4)
        # Replace deprecated append() with concat or direct assignment
    totals_with_diff = totals_df.copy()
    totals_with_diff.loc['Absolute Difference'] = diff

    # Color-code key metrics: only Test/Control rows
    color_metrics = ['conversion_rate', 'net_aov', 'orders_per_converting_visitor', 'net_sales_per_visitor']
    def highlight_metric(col):
        # col is a pandas Series of one metric across rows
        vals = col.loc[['Control','Test']]
        max_val = vals.max()
        min_val = vals.min()
        return [(
            'background-color: lightgreen' if (idx in ['Control','Test'] and v==max_val)
            else 'background-color: salmon' if (idx in ['Control','Test'] and v==min_val)
            else ''
        ) for idx, v in col.items()]

    # Style only color_metrics columns for Control/Test rows
    styled = totals_with_diff.style
    for metric in color_metrics:
        styled = styled.apply(highlight_metric, subset=[metric], axis=0)
    # Display styled dataframe
    st.dataframe(styled, use_container_width=True)

    # Statistical Tests
    obs, p_boot, ci_boot, diffs = bootstrap_rpev(df)
    z, p_z, ci_z = conversion_z_test(df)
    (u_o, p_o), (u_a, p_a) = mann_whitney_tests(df)

    # Prepare summary table
    stats_summary = pd.DataFrame([
        { 'Test': 'Revenue per Visitor (Bootstrap)', 'Statistic': f"{obs:.4f}", 'P-value': p_boot, 'CI Lower': ci_boot[0], 'CI Upper': ci_boot[1], 'Significant': 'Yes' if p_boot < 0.05 else 'No' },
        { 'Test': 'Conversion Rate (Z-test)', 'Statistic': f"{z:.4f}", 'P-value': p_z, 'CI Lower': ci_z[0], 'CI Upper': ci_z[1], 'Significant': 'Yes' if p_z < 0.05 else 'No' },
        { 'Test': 'Orders per Converter (Mann-Whitney)', 'Statistic': f"{u_o:.2f}", 'P-value': p_o, 'CI Lower': np.nan, 'CI Upper': np.nan, 'Significant': 'Yes' if p_o < 0.05 else 'No' },
        { 'Test': 'Net AOV (Mann-Whitney)', 'Statistic': f"{u_a:.2f}", 'P-value': p_a, 'CI Lower': np.nan, 'CI Upper': np.nan, 'Significant': 'Yes' if p_a < 0.05 else 'No' }
    ])

    # Calculate net sales impact and component contributions
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

        # Insight: dynamic primary contributor based on sign of impact
    contributors = {
        'Conversion Rate': contr_cr,
        'Orders per Converted Visitor': contr_opc,
        'Net AOV': contr_aov
    }
    if net_sales_impact >= 0:
        primary = max(contributors, key=lambda k: contributors[k])
    else:
        primary = min(contributors, key=lambda k: contributors[k])
    sign = 'positive' if net_sales_impact >= 0 else 'negative'
    st.write(f"**Insight:** Overall net sales impact is {sign} ({net_sales_impact:.2f}). The primary contributor is {primary}.")

    # Add impact column to summary
    stats_summary['Impact'] = [net_sales_impact, contr_cr, contr_opc, contr_aov]

    st.subheader("🔬 Statistical Tests Summary")
    st.table(stats_summary.set_index('Test'))

    # Distribution & Boxplots
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

    # Level Metrics
    shop_metrics = compute_bucket_metrics_by_level(df, 'shop')
    device_metrics = compute_bucket_metrics_by_level(df, 'device_platform')
    shop_pivot = pivot_metrics(shop_metrics, 'shop').sort_values('total_visitors_Test', ascending=False)
    device_pivot = pivot_metrics(device_metrics, 'device_platform').sort_values('total_visitors_Test', ascending=False)

    # Shop-Level Metrics
    st.subheader("🛒 Shop-Level Metrics")
    st.dataframe(shop_pivot.reset_index(drop=True), use_container_width=True)

    # Device-Level Metrics
    st.subheader("📱 Device-Level Metrics")
    st.dataframe(device_pivot.reset_index(drop=True), use_container_width=True)

    # Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Shop-Level Visuals")
        show_visuals(shop_pivot, 'shop')
    with col2:
        st.subheader("📊 Device-Level Visuals")
        show_visuals(device_pivot, 'device_platform')

if __name__ == "__main__":
    main()
