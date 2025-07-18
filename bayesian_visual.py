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

# Only orders that are both successful and have order_id > 1
df_success = df[(df['order_status'].isin(['L','O'])) & (df['order_id'] > 1)]

# All exposed visitors (per bucket)
all_visitors = (
    df.groupby(['buckets', 'exposed_visitor_id'], as_index=False)
    .size()  # just to get all unique combos
    .rename(columns={'size':'dummy'})
)

# Aggregate metrics for those who actually converted
converted_metrics = (
    df_success.groupby(['buckets', 'exposed_visitor_id'], as_index=False)
    .agg({'net_sales':'sum', 'cm1':'sum', 'cm2':'sum'})
)

# Merge: all visitors, assign sums if converted, else 0
agg = pd.merge(
    all_visitors,
    converted_metrics,
    on=['buckets', 'exposed_visitor_id'],
    how='left'
)
for col in ['net_sales', 'cm1', 'cm2']:
    agg[col] = agg[col].fillna(0)

# Per-visitor share metrics (only for those with net_sales > 0)
agg['cm1_share'] = np.where(agg['net_sales']>0, agg['cm1']/agg['net_sales'], np.nan)
agg['cm2_share'] = np.where(agg['net_sales']>0, agg['cm2']/agg['net_sales'], np.nan)

metrics = [
    ("net_sales", "NetSalesPerExposedVisitor"),
    ("cm1", "CM1PerExposedVisitor"),
    ("cm2", "CM2PerExposedVisitor"),
    ("cm1_share", "CM1 Share of NetSales (for converters only)"),
    ("cm2_share", "CM2 Share of NetSales (for converters only)")
]

# 1. Group stats table (mean, std, N for all exposed visitors)
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

# 2. Bayesian evaluation
def bayesian_diff(test, ctrl, n_draws=100000, seed=42):
    rng = np.random.default_rng(seed)
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

# 3. Display group statistics table (all exposed visitors)
fig, ax = plt.subplots(figsize=(10, 2 + 0.22*len(group_stats_df)))
ax.axis('off')
tbl = ax.table(
    cellText=group_stats_df.values,
    colLabels=group_stats_df.columns,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.auto_set_column_width(col=list(range(len(group_stats_df.columns))))
#plt.tight_layout()
#plt.show()

# 4. Display Bayesian results table
fig, ax = plt.subplots(figsize=(11, 0.8 + 0.5*len(results_df)))
ax.axis('off')
tbl = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.auto_set_column_width(col=list(range(len(results_df.columns))))
#plt.tight_layout()
#plt.show()

# 5. Posterior plots
valid_posteriors = [(name, post_diff, ci) for name, post_diff, ci in posteriors if post_diff is not None]
n_plots = len(valid_posteriors)
fig, axes = plt.subplots(n_plots, 1, figsize=(7, 2.3 * n_plots))
if n_plots == 1:
    axes = [axes]
for ax, (name, post_diff, ci) in zip(axes, valid_posteriors):
    ax.hist(post_diff, bins=50, color='skyblue', alpha=0.7, density=True)
    ax.axvline(0, color='red', linestyle='--', label='No Effect')
    ax.axvline(ci[0], color='black', linestyle=':', label='95% CI Lower')
    ax.axvline(ci[1], color='black', linestyle=':', label='95% CI Upper')
    ax.set_title(f'Posterior: {name} (Test-Control)')
    ax.set_xlabel('Difference (Test - Control)')
    ax.set_ylabel('Density')
    ax.legend()
plt.tight_layout()
plt.show()
