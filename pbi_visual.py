# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(buckets, exposed_visitor_id, net_sales, order_id, order_status)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Power BI injects your selected fields as a DataFrame named dataset
df = dataset

# --- 1. Bootstrap RPV ---
df_lo = df[df['order_status'].isin(['L','O'])]
visitor_sales = (
    df_lo
    .groupby(['buckets','exposed_visitor_id'], as_index=False)['net_sales']
    .sum()
)
test_vals = visitor_sales.loc[visitor_sales['buckets']=='Test','net_sales'].values
ctrl_vals = visitor_sales.loc[visitor_sales['buckets']=='Control','net_sales'].values
obs = test_vals.mean() - ctrl_vals.mean()

rng = np.random.default_rng(42)
diffs = np.array([
    rng.choice(test_vals, size=len(test_vals), replace=True).mean()
  - rng.choice(ctrl_vals, size=len(ctrl_vals), replace=True).mean()
  for _ in range(10000)
])
p_boot = np.mean(np.abs(diffs) >= abs(obs))
ci_boot = np.percentile(diffs, [2.5, 97.5])

# --- 2. Conversion‐rate Z‐test ---
df['converted'] = (df['order_id'] > 0).astype(int)
summ = df.groupby('buckets')['converted'].agg(total=('sum'), n=('count'))
succ = np.array([summ.loc['Test','total'], summ.loc['Control','total']])
nobs = np.array([summ.loc['Test','n'], summ.loc['Control','n']])
z_stat, p_z = proportions_ztest(succ, nobs)
p_pool = succ.sum() / nobs.sum()
se = np.sqrt(p_pool * (1-p_pool) * (1/nobs[0] + 1/nobs[1]))
z_alpha = norm.ppf(1 - 0.05/2)
diff_p = (succ[0]/nobs[0]) - (succ[1]/nobs[1])
ci_z = (diff_p - z_alpha*se, diff_p + z_alpha*se)

# --- 3. Mann‐Whitney tests ---
vs = (
    df_lo
    .groupby(['buckets','exposed_visitor_id'], as_index=False)
    .agg(total_sales=('net_sales','sum'),
         orders=('order_id','nunique'))
    .assign(
       net_aov=lambda d: d['total_sales'] / d['orders'],
       orders_per=lambda d: d['orders']
    )
)
t_o = vs.loc[vs['buckets']=='Test','orders_per']
c_o = vs.loc[vs['buckets']=='Control','orders_per']
t_a = vs.loc[vs['buckets']=='Test','net_aov']
c_a = vs.loc[vs['buckets']=='Control','net_aov']
u_o, p_o = mannwhitneyu(t_o, c_o, alternative='two-sided')
u_a, p_a = mannwhitneyu(t_a, c_a, alternative='two-sided')

# --- Build summary DataFrame ---
summary = pd.DataFrame({
    'Test': [
      'Net Revenue/Sales per Visitor (Bootstrap)',
      'Conversion Rate (Z-test)',
      'Orders per Converter (Mann-Whitney)',
      'Net AOV (Mann-Whitney)'
    ],
    'Statistic': [
      f"{obs:.3f}",
      f"{z_stat:.3f}",
      f"{u_o:.0f}",
      f"{u_a:.0f}"
    ],
    'P-value': [f"{p_boot:.5f}",f"{p_z:.5f}", f"{p_o:.5f}", f"{p_a:.5f}"],
    'CI Lower': [f"{ci_boot[0]:.5f}", f"{ci_z[0]:.5f}", np.nan, np.nan],
    'CI Upper': [f"{ci_boot[1]:.5f}", f"{ci_z[1]:.5f}", np.nan, np.nan],
    'Significant': [
      'Yes' if p_boot<0.05 else 'No',
      'Yes' if p_z<0.05 else 'No',
      'Yes' if p_o<0.05 else 'No',
      'Yes' if p_a<0.05 else 'No'
    ]
})

# --- Render as a matplotlib table ---
fig, ax = plt.subplots(figsize=(10, 1 + 0.20*len(summary)))
ax.axis('off')

tbl = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc='center',
    loc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(0.92, 0.90)

plt.tight_layout()
plt.show()
