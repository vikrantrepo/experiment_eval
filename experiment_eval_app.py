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
    required = {'buckets', 'exposed_visitor_id', 'net_sales', 'order_id', 'cm1', 'cm2', 'order_status', 'device_platform', 'shop'}
    missing = required.difference(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df['net_sales'] = df['net_sales'].fillna(0.0)
    df['order_id'] = df['order_id'].fillna(0).astype(int)
    df['order_status'] = df['order_status'].fillna('Unknown').astype(str)
    df['cm1'] = df['cm1'].fillna(0.0)
    df['cm2'] = df['cm2'].fillna(0.0)
    return df

# -------------------- METRICS FUNCTIONS --------------------
def compute_bucket_metrics(grp: pd.core.groupby.DataFrameGroupBy) -> dict:
    total_visitors = grp['exposed_visitor_id'].nunique()
    converters = grp[(grp['order_id'] > 0) & grp['order_status'].isin(['L', 'O'])]['exposed_visitor_id'].nunique()
    orders_all = grp[grp['order_id'] > 0]['order_id'].nunique()
    orders_lo = grp[grp['order_status'].isin(['L', 'O'])]['order_id'].nunique()
    sales_sum = grp['net_sales'].sum()
    cancels = grp[grp['order_status'] == 'S']['order_id'].nunique()

    # CM1 and CM2 metrics
    cm1_sum = grp['cm1'].sum()
    cm2_sum = grp['cm2'].sum()
    cm1_pv = round(cm1_sum / total_visitors, 2) if total_visitors else 0.0
    cm2_pv = round(cm2_sum / total_visitors, 2) if total_visitors else 0.0
    share_cm1 = round(cm1_sum / sales_sum, 4) if sales_sum else 0.0
    share_cm2 = round(cm2_sum / sales_sum, 4) if sales_sum else 0.0

    denom = orders_all if orders_all > 0 else None
    return {
        'total_visitors': total_visitors,
        'converting_visitors': converters,
        'conversion_rate': round(converters / total_visitors, 4) if total_visitors else 0,
        'orders_all': orders_all,
        'orders_L_O': orders_lo,
        'net_aov': round(sales_sum / orders_lo, 4) if orders_lo else 0,
        'orders_per_converting_visitor': round(orders_lo / converters, 4) if converters else 0,
        'share_of_cancelled_orders': round(cancels / denom, 4) if denom else 0,
        'net_sales_per_visitor': round(sales_sum / total_visitors, 4) if total_visitors else 0,
        'total_net_sales': round(sales_sum, 2),
        'total_cm1': round(cm1_sum, 2),
        'total_cm2': round(cm2_sum, 2),
        'cm1_per_visitor': cm1_pv,
        'cm2_per_visitor': cm2_pv,
        'share_cm1': share_cm1,
        'share_cm2': share_cm2
    }

# -------------------- AGGREGATION --------------------
def get_bucket_totals(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for bucket, grp in df.groupby('buckets'):
        rec = compute_bucket_metrics(grp)
        rec['bucket'] = bucket
        records.append(rec)
    totals = pd.DataFrame(records).set_index('bucket')
    ordered = ['Control', 'Test']
    return totals.reindex(ordered)

def compute_bucket_metrics_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    records = []
    for (lvl_val, bucket), grp in df.groupby([level, 'buckets']):
        rec = compute_bucket_metrics(grp)
        rec[level] = lvl_val
        rec['buckets'] = bucket
        records.append(rec)
    return pd.DataFrame(records).sort_values([level, 'buckets'])

# -------------------- PIVOT & DIFFERENCES --------------------
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
def bootstrap_rpev(df: pd.DataFrame, n_iters=10000, seed: int = 42):
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

def conversion_z_test(df: pd.DataFrame, alpha: float = 0.05):
    df['converted'] = df['order_id'] > 0
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    successes = np.array([summary.loc['Test','sum'], summary.loc['Control','sum']])
    nobs = np.array([summary.loc['Test','count'], summary.loc['Control','count']])
    p_pool = successes.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1]))
    z = (successes[0]/nobs[0] - successes[1]/nobs[1]) / se
    _, p = proportions_ztest(successes, nobs)
    z_alpha = norm.ppf(1 - alpha/2)
    ci = ((successes[0]/nobs[0] - successes[1]/nobs[1]) - z_alpha*se,
          (successes[0]/nobs[0] - successes[1]/nobs[1]) + z_alpha*se)
    return z, p, ci

def mann_whitney_tests(df: pd.DataFrame):
    df_lo = df[df['order_status'].isin(['L','O'])]
    visitor = df_lo.groupby(['buckets','exposed_visitor_id']).agg(
        total_sales=('net_sales','sum'),
        order_count=('order_id','nunique')
    ).assign(
        net_aov=lambda x: x.total_sales/x.order_count,
        orders_per_converted=lambda x: x.order_count
    ).reset_index()
    t_o = visitor.loc[visitor.buckets=='Test','orders_per_converted']
    c_o = visitor.loc[visitor.buckets=='Control','orders_per_converted']
    t_a = visitor.loc[visitor.buckets=='Test','net_aov']
    c_a = visitor.loc[visitor.buckets=='Control','net_aov']
    u_o, p_o = mannwhitneyu(t_o,c_o,alternative='two-sided')
    u_a, p_a = mannwhitneyu(t_a,c_a,alternative='two-sided')
    return (u_o,p_o),(u_a,p_a)

# -------------------- VISUALIZATION HELPERS --------------------
def show_visuals(df: pd.DataFrame, index_col: str):
    cols=['conversion_rate_diff_bps','net_sales_per_visitor_abs_diff','net_aov_rel_diff','orders_per_converter_rel_diff']
    sorted_df=df.sort_values('total_visitors_Test',ascending=False)
    for col in cols:
        base=alt.Chart(sorted_df).encode(
            x=alt.X(index_col,sort=list(sorted_df[index_col])),
            y=alt.Y(col,title=col.replace('_',' ').title()),
            tooltip=[index_col,col]
        )
        bars=base.mark_bar()
        fmt=".0f" if col=='conversion_rate_diff_bps' else ".2f"
        text=base.mark_text(align='center',baseline='bottom',dy=-4,fontSize=12).encode(
            text=alt.Text(col,format=fmt),
            color=alt.condition(alt.datum[col]<0,alt.value('red'),alt.value('green'))
        )
        st.altair_chart((bars+text).properties(height=300,width={'step':80}),use_container_width=True)

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title='Experiment Dashboard',layout='wide')
    st.title('📊 Experiment Results')
    path=st.file_uploader('Upload CSV',type='csv')
    if not path:
        st.info('Please upload your experiment CSV file.')
        return
    df=load_and_clean(path)

    # Filters
    with st.expander('🔍 Filter Options',expanded=False):
        shops=sorted(df['shop'].unique())
        devs=sorted(df['device_platform'].unique())
        sel_shops=st.multiselect('Shops',shops,default=shops)
        sel_devs=st.multiselect('Devices',devs,default=devs)
    df=df[df['shop'].isin(sel_shops)&df['device_platform'].isin(sel_devs)]

    # Outlier Removal
    df_lo_overall=df[df['order_status'].isin(['L','O'])]
    visitor_stats_all=df_lo_overall.groupby('exposed_visitor_id').agg(total_sales=('net_sales','sum'),order_count=('order_id','nunique')).assign(net_aov=lambda x:x.total_sales/x.order_count,orders_per_converted=lambda x:x.order_count)
    aov_cutoff=visitor_stats_all['net_aov'].quantile(0.999)
    opc_cutoff=visitor_stats_all['orders_per_converted'].quantile(0.999)
    outlier_ids=visitor_stats_all.loc[(visitor_stats_all['net_aov']>aov_cutoff)|(visitor_stats_all['orders_per_converted']>opc_cutoff)].index
    df=df[~df['exposed_visitor_id'].isin(outlier_ids)]
    excluded_counts=df_lo_overall[df_lo_overall['exposed_visitor_id'].isin(outlier_ids)].drop_duplicates(['exposed_visitor_id','buckets']).groupby('buckets')['exposed_visitor_id'].nunique().reindex(['Control','Test'],fill_value=0)
    st.write(f"**Excluded Visitors:** Control: {excluded_counts.loc['Control']}, Test: {excluded_counts.loc['Test']} (all above 99.9th percentile)")

    # Overall Metrics
    st.subheader('🏁 Overall Metrics by Bucket')
    totals_df=get_bucket_totals(df)
    diff=pd.Series(index=totals_df.columns,name='Absolute Difference')
    diff['conversion_rate']=f"{int((totals_df.loc['Test','conversion_rate']-totals_df.loc['Control','conversion_rate'])*10000)} bps"
    diff['net_aov']=round(totals_df.loc['Test','net_aov']-totals_df.loc['Control','net_aov'],4)
    diff['orders_per_converting_visitor']=round(totals_df.loc['Test','orders_per_converting_visitor']-totals_df.loc['Control','orders_per_converting_visitor'],4)
    diff['net_sales_per_visitor']=round(totals_df.loc['Test','net_sales_per_visitor']-totals_df.loc['Control','net_sales_per_visitor'],4)
    diff['total_cm1']=round(totals_df.loc['Test','total_cm1']-totals_df.loc['Control','total_cm1'],2)
    diff['total_cm2']=round(totals_df.loc['Test','total_cm2']-totals_df.loc['Control','total_cm2'],2)
    diff['cm1_per_visitor']=round(totals_df.loc['Test','cm1_per_visitor']-totals_df.loc['Control','cm1_per_visitor'],2)
    diff['cm2_per_visitor']=round(totals_df.loc['Test','cm2_per_visitor']-totals_df.loc['Control','cm2_per_visitor'],2)
    diff['share_cm1']=round(totals_df.loc['Test','share_cm1']-totals_df.loc['Control','share_cm1'],4)
    diff['share_cm2']=round(totals_df.loc['Test','share_cm2']-totals_df.loc['Control','share_cm2'],4)
    totals_with_diff=totals_df.copy()
    totals_with_diff.loc['Absolute Difference']=diff
    fmt=
    {
        'total_visitors':'{:,.0f}',
        'converting_visitors':'{:,.0f}',
        'orders_all':'{:,.0f}',
        'orders_L_O':'{:,.0f}',
        'total_net_sales':'€{:,.0f}',
        'conversion_rate':lambda v:f"{v:.2%}" if isinstance(v,(int,float,np.floating)) else v,
        'net_aov':lambda v:f"€{v:.2f}",
        'orders_per_converting_visitor':'{:.4f}',
        'share_of_cancelled_orders':'{:.2%}',
        'net_sales_per_visitor':lambda v:f"€{v:.2f}" if isinstance(v,(int,float,np.floating)) else v,
        'total_cm1':lambda v:f"€{v:.2f}",
        'total_cm2':lambda v:f"€{v:.2f}",
        'cm1_per_visitor':lambda v:f"€{v:.2f}" if isinstance(v,(int,float,np.floating)) else v,
        'cm2_per_visitor':lambda v:f"€{v:.2f}" if isinstance(v,(int,float,np.floating)) else v,
        'share_cm1':lambda v:f"{v:.2%}" if isinstance(v,(int,float,np.floating)) else v,
        'share_cm2':lambda v:f"{v:.2%}" if isinstance(v,(int,float,np.floating)) else v
    }
    st.dataframe(totals_with_diff.style.format(fmt),use_container_width=True)

    # Statistical Tests
    obs,p_boot,ci_boot,diffs=bootstrap_rpev(df)
    z,p_z,ci_z=conversion_z_test(df)
    (u_o,p_o),(u_a,p_a)=mann_whitney_tests(df)

    stats_summary=pd.DataFrame([
        {'Test':'Revenue per Visitor (Bootstrap)','Statistic':f"{obs:.4f}",'P-value':p_boot,'CI Lower':ci_boot[0],'CI Upper':ci_boot[1],'Significant':'Yes' if p_boot<0.05 else 'No'},
        {'Test':'Conversion Rate (Z-test)','Statistic':f"{z:.4f}",'P-value':p_z,'CI Lower':ci_z[0],'CI Upper':ci_z[1],'Significant':'Yes' if p_z<0.05 else 'No'},
        {'Test':'Orders per Converter (Mann-Whitney)','Statistic':f"{u_o:.2f}",'P-value':p_o,'CI Lower':np.nan,'CI Upper':np.nan,'Significant':'Yes' if p_o<0.05 else 'No'},
        {'Test':'Net AOV (Mann-Whitney)','Statistic':f"{u_a:.2f}",'P-value':p_a,'CI Lower':np.nan,'CI Upper':np.nan,'Significant':'Yes' if p_a<0.05 else 'No'}
    ])

    tot_vis_test=totals_df.loc['Test','total_visitors']
    cr_c=totals_df.loc['Control','conversion_rate']
    opc_c=totals_df.loc['Control','orders_per_converting_visitor']
    aov_c=totals_df.loc['Control','net_aov']
    dnspv=totals_df.loc['Test','net_sales_per_visitor']-totals_df.loc['Control','net_sales_per_visitor']
    dcr=totals_df.loc['Test','conversion_rate']-cr_c
    dopc=totals_df.loc['Test','orders_per_converting_visitor']-opc_c
    daov=totals_df.loc['Test','net_aov']-aov_c
    nsi=dnspv*tot_vis_test
    ccr=dcr*opc_c*aov_c*tot_vis_test
    copc=cr_c*dopc*aov_c*tot_vis_test
    caov=cr_c*opc_c*daov*tot_vis_test

    contrib={'Conversion Rate':ccr,'Orders per Converted Visitor':copc,'Net AOV':caov}
    primary=max(contrib,key=contrib.get) if nsi>=0 else min(contrib,key=contrib.get)
    sign='positive' if nsi>=0 else 'negative'
    st.write(f"**Insight:** Overall net sales impact is {sign} ({nsi:.2f}). The primary contributor is {primary}.")

    stats_summary['Impact']=[nsi,ccr,copc,caov]
    st.subheader('🔬 Statistical Tests Summary')
    st.table(stats_summary.set_index('Test'))

    st.subheader('📈 Distribution and Boxplots')
    df_lo=df[df['order_status'].isin(['L','O'])]
    vs=df_lo.groupby(['buckets','exposed_visitor_id']).agg(total_sales=('net_sales','sum'),order_count=('order_id','nunique')).assign(net_aov=lambda x:x.total_sales/x.order_count,orders_per_converted=lambda x:x.order_count).reset_index()
    c1,c2,c3=st.columns(3)
    with c1:
        f1,a1=plt.subplots(figsize=(4,3))
        a1.hist(diffs,bins=50,alpha=0.7)
        a1.axvline(obs,linestyle='--');a1.axvline(ci_boot[0],linestyle=':');a1.axvline(ci_boot[1],linestyle=':')
        a1.set_title('Bootstrap Distribution')
        st.pyplot(f1)
    with c2:
        f2,a2=plt.subplots(figsize=(4,3))
        vs.boxplot(column='net_aov',by='buckets',ax=a2)
        a2.set_title('Net AOV by Bucket');plt.suptitle('');st.pyplot(f2)
    with c3:
        f3,a3=plt.subplots(figsize=(4,3))
        vs.boxplot(column='order_count',by='buckets',ax=a3)
        a3.set_title('Orders per Converted Visitor');plt.suptitle('');st.pyplot(f3)

    shop_m=compute_bucket_metrics_by_level(df,'shop')
    dev_m=compute_bucket_metrics_by_level(df,'device_platform')
    shop_p=pivot_metrics(shop_m,'shop').sort_values('total_visitors_Test',ascending=False)
    dev_p=pivot_metrics(dev_m,'device_platform').sort_values('total_visitors_Test',ascending=False)

    st.subheader('🛒 Shop-Level Metrics');st.dataframe(shop_p.reset_index(drop=True),use_container_width=True)
    st.subheader('📱 Device-Level Metrics');st.dataframe(dev_p.reset_index(drop=True),use_container_width=True)
    col1,col2=st.columns(2)
    with col1:
        st.subheader('📊 Shop-Level Visuals');show_visuals(shop_p,'shop')
    with col2:
        st.subheader('📊 Device-Level Visuals');show_visuals(dev_p,'device_platform')

    def compute_contribs(df,seg):
        df=df.copy()
        df['net_sales_impact']=df['net_sales_per_visitor_abs_diff']*df[f'total_visitors_Test']
        df['cr_c']=df[f'conversion_rate_Control'];df['opc_c']=df[f'orders_per_converting_visitor_Control'];df['aov_c']=df[f'net_aov_Control']
        df['delta_cr']=df[f'conversion_rate_Test']-df[f'conversion_rate_Control']
        df['delta_opc']=df[f'orders_per_converting_visitor_Test']-df[f'orders_per_converting_visitor_Control']
        df['delta_aov']=df[f'net_aov_Test']-df[f'net_aov_Control']
        df['contr_cr']=df['delta_cr']*df['opc_c']*df['aov_c']*df[f'total_visitors_Test']
        df['contr_opc']=df['cr_c']*df['delta_opc']*df['aov_c']*df[f'total_visitors_Test']
        df['contr_aov']=df['cr_c']*df['opc_c']*df['delta_aov']*df[f'total_visitors_Test']
        df['main_contributor']=df.apply(lambda r:max({'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']},key=lambda k:{'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']}[k]) if r['net_sales_impact']>=0 else min({'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']},key=lambda k:{'Conversion Rate':r['contr_cr'],'Orders per Converted Visitor':r['contr_opc'],'Net AOV':r['contr_aov']}[k]),axis=1)
        return df

    shop_imp=compute_contribs(shop_p,'shop')
    dev_imp=compute_contribs(dev_p,'device_platform')
    mix=df.copy();mix['shop_device']=mix['shop']+' | '+mix['device_platform']
    mix_p=pivot_metrics(compute_bucket_metrics_by_level(mix,'shop_device'),'shop_device').sort_values('total_visitors_Test',ascending=False)
    mix_imp=compute_contribs(mix_p,'shop_device')

    insights=[]
    for n,imp,c in [('Shop',shop_imp,'shop'),('Device',dev_imp,'device_platform'),('Shop & Device',mix_imp,'shop_device')]:
        best=imp.nlargest(1,'net_sales_impact');worst=imp.nsmallest(1,'net_sales_impact')
        insights.append(f"**{n}**: Best segment '{best.iloc[0][c]}' impact {best.iloc[0]['net_sales_impact']:.2f} ({best.iloc[0]['main_contributor']}); Worst '{worst.iloc[0][c]}' impact {worst.iloc[0]['net_sales_impact']:.2f} ({worst.iloc[0]['main_contributor']}).")
    st.markdown('**Segment Impact Insights:**')
    for i in insights:st.markdown(f'- {i}')

    with st.expander('📌 Segment Impact Analysis',expanded=False):
        st.subheader('Shop Segments');st.table(shop_imp.set_index('shop')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])
        st.subheader('Device Segments');st.table(dev_imp.set_index('device_platform')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])
        st.subheader('Shop & Device Mix Segments');st.table(mix_imp.set_index('shop_device')[['net_sales_impact','contr_cr','contr_opc','contr_aov','main_contributor']])

if __name__ == '__main__': main()
