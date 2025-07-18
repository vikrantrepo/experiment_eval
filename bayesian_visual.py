import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df = dataset.copy()

# Only successful orders with order_id > 1
df_success = df[(df['order_status'].isin(['L','O'])) & (df['order_id'] > 1)]

# All exposed visitors (per bucket)
all_visitors = (
    df.groupby(['buckets', 'exposed_visitor_id'], as_index=False)
    .size()
    .rename(columns={'size':'dummy'})
)

# Aggregate metrics for converters
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

# Group stats table (mean, std, N for all exposed visitors)
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

# Bayesian evaluation
def bayesian_diff(test, ctrl, n_draws=15000):
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

# ========== IMPROVED LAYOUT ==========

import matplotlib.gridspec as gridspec

n_post = len([x for x in posteriors if x[1] is not None])
fig_height = 4.6 + 2.2 * n_post  # More space per plot
fig_width = 30

gs = gridspec.GridSpec(
    n_post, 2,
    width_ratios=[2.4, 2.6],
    height_ratios=[1.25]*n_post,
    wspace=0.22, hspace=0.55
)

# LEFT: Both tables, stacked
table_ax = plt.subplot(gs[:,0])
table_ax.axis('off')

# Table 1: group stats (at top)
table1 = table_ax.table(
    cellText=group_stats_df.round(4).astype(str).values,
    colWidths=[0.32, 0.13, 0.13, 0.13, 0.13, 0.13],
    colLabels=group_stats_df.columns,
    loc='upper center',
    cellLoc='center',
    bbox=[0, 0.62, 1, 0.36]
)
table1.auto_set_font_size(False)
table1.set_fontsize(11)
for key, cell in table1.get_celld().items():
    cell.set_linewidth(0.6)
    cell.set_fontsize(10)
    cell.set_clip_on(False)

# Table 2: bayesian results (below)
table2 = table_ax.table(
    cellText=results_df.astype(str).values,
    colWidths=[0.32, 0.13, 0.13, 0.13, 0.13, 0.13],
    colLabels=results_df.columns,
    loc='lower center',
    cellLoc='center',
    bbox=[0, 0.06, 1, 0.54]
)
table2.auto_set_font_size(False)
table2.set_fontsize(11)
for key, cell in table2.get_celld().items():
    cell.set_linewidth(0.6)
    cell.set_fontsize(10)
    cell.set_clip_on(False)

table_ax.set_title('Group & Bayesian Tables', pad=14, fontsize=13)

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
    if i == 0:
        ax.legend(fontsize=9, loc='upper right')

plt.gcf().set_size_inches(fig_width, fig_height)
plt.tight_layout()
plt.show()
