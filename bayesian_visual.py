# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(buckets, net_sales, exposed_visitor_id)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df = dataset.copy()

df_success = df[(df['order_status'].isin(['L','O'])) & (df['order_id'] > 1)]
all_visitors = (
    df.groupby(['buckets', 'exposed_visitor_id'], as_index=False)
    .size()
    .rename(columns={'size':'dummy'})
)
converted_metrics = (
    df_success.groupby(['buckets', 'exposed_visitor_id'], as_index=False)
    .agg({'net_sales':'sum', 'cm1':'sum', 'cm2':'sum'})
)
agg = pd.merge(
    all_visitors,
    converted_metrics,
    on=['buckets', 'exposed_visitor_id'],
    how='left'
)
for col in ['net_sales', 'cm1', 'cm2']:
    agg[col] = agg[col].fillna(0)
agg['cm1_share'] = np.where(agg['net_sales']>0, agg['cm1']/agg['net_sales'], np.nan)
agg['cm2_share'] = np.where(agg['net_sales']>0, agg['cm2']/agg['net_sales'], np.nan)

metrics = [
    ("net_sales", "NetSalesPerExposedVisitor"),
    ("cm1", "CM1PerExposedVisitor"),
    ("cm2", "CM2PerExposedVisitor"),
    ("cm1_share", "CM1 Share of NetSales (for converters only)"),
    ("cm2_share", "CM2 Share of NetSales (for converters only)")
]

group_stats = []
for col, label in metrics:
    for group in ['Test', 'Control']:
        vals = agg.loc[agg['buckets'] == group, col].dropna()
        group_stats.append({
            'Metric': label,
            'Group': group,
            'Mean': np.round(np.mean(vals), 4) if len(vals)>0 else 'NA',
            'Std': np.round(np.std(vals, ddof=1), 4) if len(vals)>1 else 'NA',
            'N': len(vals)
        })
group_stats_df = pd.DataFrame(group_stats)

def bayesian_diff(test, ctrl, n_draws=150000):
    rng = np.random.default_rng()
    m_t, s_t, n_t = np.mean(test), np.std(test, ddof=1), len(test)
    m_c, s_c, n_c = np.mean(ctrl), np.std(ctrl, ddof=1), len(ctrl)
    s_t = s_t if s_t > 0 else 1e-9
    s_c = s_c if s_c > 0 else 1e-9
    if n_t < 2 or n_c < 2:
        return np.nan, [np.nan, np.nan], np.nan, None
    post_test = rng.normal(m_t, s_t/np.sqrt(n_t), n_draws)
    post_ctrl = rng.normal(m_c, s_c/np.sqrt(n_c), n_draws)
    post_diff = post_test - post_ctrl
    ci = np.percentile(post_diff, [2.5, 97.5])
    prob = np.mean(post_diff > 0)
    mean = np.mean(post_diff)
    return mean, ci, prob, post_diff

results = []
posteriors = []
for col, label in metrics:
    test = agg.loc[agg['buckets'] == 'Test', col].dropna().values
    ctrl = agg.loc[agg['buckets'] == 'Control', col].dropna().values
    mean, ci, prob, post_diff = bayesian_diff(test, ctrl)
    results.append({
        "Metric": label + " (Bayesian)",
        "Mean Diff": round(mean, 4) if pd.notnull(mean) else "NA",
        "CI Lower": round(ci[0], 4) if pd.notnull(ci[0]) else "NA",
        "CI Upper": round(ci[1], 4) if pd.notnull(ci[1]) else "NA",
        "P(Test>Control)": f"{prob:.2%}" if pd.notnull(prob) else "NA",
        "Significant": "Yes" if pd.notnull(prob) and prob > 0.95 else "No"
    })
    posteriors.append((label, post_diff, ci))
results_df = pd.DataFrame(results)

# IMPROVED LAYOUT
n_post = len([x for x in posteriors if x[1] is not None])
fig_height = 2.5 * max(n_post, 3)
fig_width = 20
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(
    n_post, 2,
    width_ratios=[1.2, 2.5],
    height_ratios=[1]*n_post,
    wspace=0.15, hspace=0.28
)

# LEFT: Both tables, stacked
table_ax = plt.subplot(gs[:,0])
table_ax.axis('off')

# Table 1: group stats (at top)
table1 = table_ax.table(
    cellText=group_stats_df.values,
    colLabels=group_stats_df.columns,
    loc='upper center',
    cellLoc='center',
    bbox=[0, 0.54, 1, 0.43]
)
table1.auto_set_font_size(False)
table1.set_fontsize(11)

# Table 2: bayesian results (below)
table2 = table_ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='lower center',
    cellLoc='center',
    bbox=[0, 0.04, 1, 0.47]
)
table2.auto_set_font_size(False)
table2.set_fontsize(11)

table_ax.set_title('Group & Bayesian Tables', pad=16, fontsize=13)

# RIGHT: Posterior plots stacked
valid_posteriors = [(name, post_diff, ci) for name, post_diff, ci in posteriors if post_diff is not None]
for i, (name, post_diff, ci) in enumerate(valid_posteriors):
    ax = plt.subplot(gs[i,1])
    ax.hist(post_diff, bins=40, color='skyblue', alpha=0.7, density=True)
    ax.axvline(0, color='red', linestyle='--', label='No Effect')
    ax.axvline(ci[0], color='black', linestyle=':', label='95% CI Lower')
    ax.axvline(ci[1], color='black', linestyle=':', label='95% CI Upper')
    ax.set_title(f'Posterior: {name} (Test-Control)', fontsize=12)
    ax.set_xlabel('Difference (Test - Control)')
    ax.set_ylabel('Density')
    if i==0:
        ax.legend()

plt.tight_layout()
plt.show()
