import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
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
    return totals.reindex(['Control', 'Test'])

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
        rng.choice(test, size=len(test), replace=True).mean() - rng.choice(ctrl, size=len(ctrl), replace=True).mean()
        for _ in range(n_iters)
    ])
    p_val = np.mean(np.abs(diffs) >= abs(obs))
    ci = np.percentile(diffs, [2.5, 97.5])
    return obs, p_val, ci, diffs

def conversion_z_test(df: pd.DataFrame, alpha=0.05):
    df['converted'] = df['order_id'] > 0
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    successes = np.array([summary.loc['Test','sum'], summary.loc['Control','sum']])
    nobs = np.array([summary.loc['Test','count'], summary.loc['Control','count']])
    p_pool = successes.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1-p_pool) * (1/nobs[0] + 1/nobs[1]))
    diff = (successes[0]/nobs[0]) - (successes[1]/nobs[1])
    z = diff / se
    _, p = proportions_ztest(successes, nobs)
    z_alpha = norm.ppf(1-alpha/2)
    ci = (diff - z_alpha*se, diff + z_alpha*se)
    return z, p, ci

def mann_whitney_tests(df: pd.DataFrame):
    df_lo = df[df['order_status'].isin(['L','O'])]
    visitor = df_lo.groupby(['buckets','exposed_visitor_id']).agg(
        total_sales=('net_sales','sum'), order_count=('order_id','nunique')
    ).assign(
        net_aov=lambda x: x.total_sales/x.order_count,
        orders_per_converted=lambda x: x.order_count
    ).reset_index()
    t_o = visitor.query("buckets=='Test'")['orders_per_converted']
    c_o = visitor.query("buckets=='Control'")['orders_per_converted']
    t_a = visitor.query("buckets=='Test'")['net_aov']
    c_a = visitor.query("buckets=='Control'")['net_aov']
    u_o, p_o = mannwhitneyu(t_o, c_o, alternative='two-sided')
    u_a, p_a = mannwhitneyu(t_a, c_a, alternative='two-sided')
    return (u_o, p_o), (u_a, p_a)

# -------------------- VISUALIZATION HELPERS --------------------
def show_visuals(df: pd.DataFrame, index_col: str):
    cols = ['conversion_rate_diff_bps', 'net_sales_per_visitor_abs_diff', 'net_aov_rel_diff', 'orders_per_converter_rel_diff']
    sorted_df = df.sort_values(f'total_visitors_Test', ascending=False)
    for col in cols:
        if col not in sorted_df.columns:
            continue
        st.write(f"**{col.replace('_', ' ').title()}**")
        base = alt.Chart(sorted_df).encode(
            x=alt.X(
                index_col,
                sort=list(sorted_df[index_col]),
                axis=alt.Axis(
                    labelAngle=-45,
                    labelAlign='right',
                    labelLimit=200,
                    labelOverlap='greedy'
                )
            ),
            y=alt.Y(
                col,
                title=col.replace('_', ' ').title()
            ),
            tooltip=[index_col, col]
        )
        bars = base.mark_bar()
        fmt = (
            ".0f" if col == 'conversion_rate_diff_bps' else
            ".1%" if col in ['net_aov_rel_diff', 'orders_per_converter_rel_diff'] else
            ".2f"
        )
        text = base.mark_text(
            align='center',
            baseline='bottom',
            dy=-4,
            fontSize=12
        ).encode(
            text=alt.Text(col, format=fmt),
            color=alt.condition(
                alt.datum[col] < 0,
                alt.value('red'),
                alt.value('green')
            )
        )
        chart = (bars + text).properties(
            height=300,
            width={'step': 80}
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(chart, use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="Experiment Dashboard", layout="wide")
    st.title("📊 Experiment Results")
    path = st.file_uploader("Upload CSV",type='csv')
    if not path:
        st.info("Please upload your experiment CSV file.")
        return
    df = load_and_clean(path)

    # Filters
    with st.expander("🔍 Filter Options",expanded=False):
        shops, devs = sorted(df['shop'].unique()), sorted(df['device_platform'].unique())
        sel_shops = st.multiselect("Shops",shops,default=shops)
        sel_devs = st.multiselect("Devices",devs,default=devs)
    df = df[df['shop'].isin(sel_shops)&df['device_platform'].isin(sel_devs)]

    # Outlier removal
    df_lo = df[df['order_status'].isin(['L','O'])]
    stats = df_lo.groupby('exposed_visitor_id').agg(total_sales=('net_sales','sum'),order_count=('order_id','nunique')).assign(
        net_aov=lambda x: x.total_sales/x.order_count, orders_per_converted=lambda x: x.order_count
    )
    aov_cut, opc_cut = stats['net_aov'].quantile(0.999), stats['orders_per_converted'].quantile(0.999)
    outliers = stats[(stats['net_aov']>aov_cut)|(stats['orders_per_converted']>opc_cut)].index
    df = df[~df['exposed_visitor_id'].isin(outliers)]
    excl = df_lo[df_lo['exposed_visitor_id'].isin(outliers)][['exposed_visitor_id','buckets']].drop_duplicates()
    counts = excl.groupby('buckets')['exposed_visitor_id'].nunique().reindex(['Control','Test'],fill_value=0)
    st.write(f"**Excluded Visitors:** Control: {counts['Control']}, Test: {counts['Test']} (above 99.9th pct)")

    # Overall metrics
    st.subheader("🏁 Overall Metrics by Bucket")
    totals = get_bucket_totals(df)
    diff = pd.Series(name='Absolute Difference',index=totals.columns)
    diff['conversion_rate'] = f"{int(round((totals.loc['Test','conversion_rate']-totals.loc['Control','conversion_rate'])*10000))} bps"
    diff['net_aov'] = round(totals.loc['Test','net_aov']-totals.loc['Control','net_aov'],4)
    diff['orders_per_converting_visitor'] = round(totals.loc['Test','orders_per_converting_visitor']-totals.loc['Control','orders_per_converting_visitor'],4)
    diff['net_sales_per_visitor'] = round(totals.loc['Test','net_sales_per_visitor']-totals.loc['Control','net_sales_per_visitor'],4)
    df_tot = totals.copy(); df_tot.loc['Absolute Difference'] = diff

    # Style totals
    def highlight(col):
        vals=col.loc[['Control','Test']]; return [ 'background-color: lightgreen' if (idx in ['Control','Test'] and v==vals.max())
            else 'background-color: salmon' if (idx in ['Control','Test'] and v==vals.min()) else '' for idx,v in col.items()]
    styled = df_tot.style
    for m in ['conversion_rate','net_aov','orders_per_converting_visitor','net_sales_per_visitor']:
        styled=styled.apply(highlight,subset=[m],axis=0)
    fmt = {
        'total_visitors':'{:,.0f}','converting_visitors':'{:,.0f}','orders_all':'{:,.0f}','orders_L_O':'{:,.0f}',
        'total_net_sales':'€{:,.0f}', 'conversion_rate':lambda v: f"{v:.2%}" if isinstance(v,(int,float,np.floating)) else v,
        'net_aov':lambda v: f"€{v:.2f}", 'orders_per_converting_visitor':'{:.4f}', 'share_of_cancelled_orders':'{:.2%}',
        'net_sales_per_visitor':lambda v: f"€{v:,.0f}" if isinstance(v,(int,float,np.floating)) else v
    }
    styled=styled.format(fmt)
    st.dataframe(styled,use_container_width=True)

    # Stats tests
    obs,pb,ci,diffs=bootstrap_rpev(df)
    z,pz,ci_z=conversion_z_test(df)
    (u_o,p_o),(u_a,p_a)=mann_whitney_tests(df)

    # Summary table
    stats_summary=pd.DataFrame([
        {'Test':'Revenue per Visitor (Bootstrap)','Statistic':f"{obs:.4f}",'P-value':pb,'CI Lower':ci[0],'CI Upper':ci[1],'Significant':'Yes' if pb<0.05 else 'No'},
        {'Test':'Conversion Rate (Z-test)','Statistic':f"{z:.4f}",'P-value':pz,'CI Lower':ci_z[0],'CI Upper':ci_z[1],'Significant':'Yes' if pz<0.05 else 'No'},
        {'Test':'Orders per Converter (M-W)','Statistic':f"{u_o:.2f}",'P-value':p_o,'CI Lower':np.nan,'CI Upper':np.nan,'Significant':'Yes' if p_o<0.05 else 'No'},
        {'Test':'Net AOV (M-W)','Statistic':f"{u_a:.2f}",'P-value':p_a,'CI Lower':np.nan,'CI Upper':np.nan,'Significant':'Yes' if p_a<0.05 else 'No'}
    ])

    # Net sales impact
    tv=totals.loc['Test','total_visitors']; crc=totals.loc['Control','conversion_rate']; opc=totals.loc['Control','orders_per_converting_visitor']; aov=totals.loc['Control','net_aov']
    dn=totals.loc['Test','net_sales_per_visitor']-totals.loc['Control','net_sales_per_visitor']
    dc=totals.loc['Test','conversion_rate']-crc; do=totals.loc['Test','orders_per_converting_visitor']-opc; da=totals.loc['Test','net_aov']-aov
    nsi=dn*tv; cr_cont=dc*opc*aov*tv; opc_cont=crc*do*aov*tv; aov_cont=crc*opc*da*tv
    conts={'Conversion Rate':cr_cont,'Orders per Converted Visitor':opc_cont,'Net AOV':aov_cont}
    primary = max(conts, key=conts.get) if nsi>=0 else min(conts, key=conts.get)
    sign = 'positive' if nsi>=0 else 'negative'
    st.write(f"**Insight:** Overall net sales impact is {sign} (€{nsi:,.0f}). The primary contributor is {primary}.")
    stats_summary['Impact']= [nsi, cr_cont, opc_cont, aov_cont]
    stats_summary['Impact']=stats_summary['Impact'].map(lambda x: f"€{x:,.0f}")

    st.subheader("🔬 Statistical Tests Summary")
    st.table(stats_summary.set_index('Test'))

    # Distributions & boxplots
    st.subheader("📈 Distribution and Boxplots")
    df_lo2=df[df['order_status'].isin(['L','O'])]
    visitor_stats=df_lo2.groupby(['buckets','exposed_visitor_id']).agg(total_sales=('net_sales','sum'),order_count=('order_id','nunique')).assign(
        net_aov=lambda x: x.total_sales/x.order_count, orders_per_converted=lambda x: x.order_count
    ).reset_index()
    c1,c2,c3=st.columns(3)
    with c1:
        fig1,ax1=plt.subplots(figsize=(4,3))
        ax1.hist(diffs,bins=50,alpha=0.7)
        ax1.axvline(obs,linestyle='--'); ax1.axvline(ci[0],linestyle=':'); ax1.axvline(ci[1],linestyle=':')
        ax1.set_title('Bootstrap Distribution')
        st.pyplot(fig1)
    with c2:
        fig2,ax2=plt.subplots(figsize=(4,3))
        visitor_stats.boxplot(column='net_aov',by='buckets',ax=ax2)
        ax2.set_title('Net AOV by Bucket'); plt.suptitle('')
        st.pyplot(fig2)
    with c3:
        fig3,ax3=plt.subplots(figsize=(4,3))
        visitor_stats.boxplot(column='order_count',by='buckets',ax=ax3)
        ax3.set_title('Orders per Converted Visitor'); plt.suptitle('')
        st.pyplot(fig3)

    # Shop & Device metrics
    shop_m=compute_bucket_metrics_by_level(df,'shop')
    dev_m=compute_bucket_metrics_by_level(df,'device_platform')
    shop_pivot=pivot_metrics(shop_m,'shop').sort_values('total_visitors_Test',ascending=False)
    dev_pivot=pivot_metrics(dev_m,'device_platform').sort_values('total_visitors_Test',ascending=False)
    st.subheader("🛒 Shop-Level Metrics")
    styled_shop=shop_pivot.style.format({'net_sales_per_visitor_abs_diff':'€{:.0f}'})
    st.dataframe(styled_shop,use_container_width=True)
    st.subheader("📱 Device-Level Metrics")
    styled_dev=dev_pivot.style.format({'net_sales_per_visitor_abs_diff':'€{:.0f}'})
    st.dataframe(styled_dev,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.subheader("📊 Shop-Level Visuals")
        show_visuals(shop_pivot,'shop')
    with c2:
        st.subheader("📊 Device-Level Visuals")
        show_visuals(dev_pivot,'device_platform')

    # Segment Impact Analysis
    def compute_contribs(df, col):
        df2=df.copy()
        df2['net_sales_impact']=df2['net_sales_per_visitor_abs_diff']*df2[f'total_visitors_Test']
        df2['cr_c']=df2['conversion_rate_Control']; df2['opc_c']=df2['orders_per_converting_visitor_Control']; df2['aov_c']=df2['net_aov_Control']
        df2['delta_cr']=df2['conversion_rate_Test']-df2['conversion_rate_Control']
        df2['delta_opc']=df2['orders_per_converting_visitor_Test']-df2['orders_per_converting_visitor_Control']
        df2['delta_aov']=df2['net_aov_Test']-df2['net_aov_Control']
        df2['contr_cr']=df2['delta_cr']*df2['opc_c']*df2['aov_c']*df2['total_visitors_Test']
        df2['contr_opc']=df2['cr_c']*df2['delta_opc']*df2['aov_c']*df2['total_visitors_Test']
        df2['contr_aov']=df2['cr_c']*df2['opc_c']*df2['delta_aov']*df2['total_visitors_Test']
        def pick(r): return max({'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']}, key=lambda k: r['contr_' + k.split()[0].lower()]) if r['net_sales_impact']>=0 else min({'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']}, key=lambda k: r['contr_' + k.split()[0].lower()])
        df2['main_contributor']=df2.apply(pick,axis=1)
        return df2
    shop_imp=compute_contribs(shop_pivot,'shop'); dev_imp=compute_contribs(dev_pivot,'device_platform')
    mix=df.copy(); mix['shop_device']=mix['shop']+' | '+mix['device_platform']
    mix_imp=compute_contribs(pivot_metrics(compute_bucket_metrics_by_level(mix,'shop_device'),'shop_device').sort_values('total_visitors_Test',ascending=False),'shop_device')

    st.markdown("**Segment Impact Insights:**")
    for name, df_seg,col in [('Shop',shop_imp,'shop'),('Device',dev_imp,'device_platform'),('Shop & Device',mix_imp,'shop_device')]:
        best=df_seg.nlargest(1,'net_sales_impact')
        worst=df_seg.nsmallest(1,'net_sales_impact')
        st.markdown(f"- **{name}**: Best segment “{best.iloc[0][col]}” with impact €{best.iloc[0]['net_sales_impact']:,.0f} (main contributor: {best.iloc[0]['main_contributor']}); Worst segment “{worst.iloc[0][col]}” with impact €{worst.iloc[0]['net_sales_impact']:,.0f} (main contributor: {worst.iloc[0]['main_contributor']}).")

    with st.expander("📌 Segment Impact Analysis",expanded=False):
        st.subheader("Shop Segments")
        shop_table=shop_imp.sort_values('net_sales_impact',ascending=False)
        shop_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']]=shop_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']].applymap(lambda x: f"€{x:,.0f}")
        st.table(shop_table.set_index('shop')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])

        st.subheader("Device Segments")
        dev_table=dev_imp.sort_values('net_sales_impact',ascending=False)
        dev_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']]=dev_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']].applymap(lambda x: f"€{x:,.0f}")
        st.table(dev_table.set_index('device_platform')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])

        st.subheader("Shop & Device Mix Segments")
        mix_table=mix_imp.sort_values('net_sales_impact',ascending=False)
        mix_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']]=mix_table[['net_sales_impact','contr_cr','contr_opc','contr_aov']].applymap(lambda x: f"€{x:,.0f}")
        st.table(mix_table.set_index('shop_device')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])

if __name__ == "__main__":
    main()
