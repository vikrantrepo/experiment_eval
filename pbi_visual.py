# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(buckets, exposed_visitor_id, net_sales, order_id, order_status)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

df = dataset.copy()
df['net_sales'] = df['net_sales'].fillna(0.0)
df['order_id'] = df['order_id'].fillna(0).astype(int)
df['order_status'] = df['order_status'].fillna('Unknown').astype(str)

# Compute bucket metrics
def compute_bucket_metrics(grp):
    total_visitors = grp['exposed_visitor_id'].nunique()
    converters = grp[(grp['order_id'] > 1) & grp['order_status'].isin(['L', 'O'])]['exposed_visitor_id'].nunique()
    orders_lo = grp[grp['order_status'].isin(['L', 'O'])]['order_id'].nunique()
    sales_sum = grp['net_sales'].sum()
    net_aov = sales_sum / orders_lo if orders_lo else 0
    orders_per_converter = orders_lo / converters if converters else 0
    conversion_rate = converters / total_visitors if total_visitors else 0
    net_sales_per_visitor = sales_sum / total_visitors if total_visitors else 0
    return pd.Series({
        'total_visitors': total_visitors,
        'converting_visitors': converters,
        'conversion_rate': conversion_rate,
        'orders_L_O': orders_lo,
        'net_aov': net_aov,
        'orders_per_converter': orders_per_converter,
        'net_sales_per_visitor': net_sales_per_visitor,
        'total_net_sales': sales_sum
    })

totals = df.groupby('buckets').apply(compute_bucket_metrics)
totals = totals.reindex(['Control', 'Test'])

delta_nspv = totals.loc['Test','net_sales_per_visitor'] - totals.loc['Control','net_sales_per_visitor']
total_vis_test = totals.loc['Test','total_visitors']
net_sales_impact = delta_nspv * total_vis_test

cr_c = totals.loc['Control','conversion_rate']
opc_c = totals.loc['Control','orders_per_converter']
aov_c = totals.loc['Control','net_aov']

delta_cr = totals.loc['Test','conversion_rate'] - cr_c
delta_opc = totals.loc['Test','orders_per_converter'] - opc_c
delta_aov = totals.loc['Test','net_aov'] - aov_c

contr_cr  = delta_cr  * opc_c * aov_c * total_vis_test
contr_opc = cr_c * delta_opc * aov_c * total_vis_test
contr_aov = cr_c * opc_c * delta_aov * total_vis_test

# Statistical tests
def bootstrap_rpev(df: pd.DataFrame, n_iters=20000):
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
    return obs, p_val, ci

def conversion_z_test(df: pd.DataFrame, alpha=0.05):
    df['converted'] = df['order_id'] > 1
    summary = df.groupby('buckets')['converted'].agg(['sum', 'count'])
    successes = np.array([summary.loc['Test', 'sum'], summary.loc['Control', 'sum']])
    nobs = np.array([summary.loc['Test', 'count'], summary.loc['Control', 'count']])
    _, p = proportions_ztest(successes, nobs)
    p_pool = successes.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs[0] + 1/nobs[1]))
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

obs, p_boot, ci_boot = bootstrap_rpev(df)
z, p_z, ci_z = conversion_z_test(df)
(u_o, p_o), (u_a, p_a) = mann_whitney_tests(df)

# Prepare summary table (without Main Contributor)
table_data = [
    ["Revenue per Visitor (Bootstrap)", f"{obs:.4f}", f"{p_boot:.3f}", f"{ci_boot[0]:.2f}", f"{ci_boot[1]:.2f}", "Yes" if p_boot < 0.05 else "No", f"{net_sales_impact:,.2f}"],
    ["Conversion Rate (Z-test)", f"{z:.4f}", f"{p_z:.3f}", f"{ci_z[0]:.4f}", f"{ci_z[1]:.4f}", "Yes" if p_z < 0.05 else "No", f"{contr_cr:,.2f}"],
    ["Orders per Converter (Mann-Whitney)", f"{u_o:.2f}", f"{p_o:.3f}", "-", "-", "Yes" if p_o < 0.05 else "No", f"{contr_opc:,.2f}"],
    ["Net AOV (Mann-Whitney)", f"{u_a:.2f}", f"{p_a:.3f}", "-", "-", "Yes" if p_a < 0.05 else "No", f"{contr_aov:,.2f}"]
]

columns = [
    "Test",
    "Statistic",
    "P-value",
    "CI Lower",
    "CI Upper",
    "Significant",
    "Net Sales Impact"
]

# Plot as Matplotlib table
fig, ax = plt.subplots(figsize=(11, 2.5))
ax.axis('off')
tbl = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='left'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.auto_set_column_width([i for i in range(len(columns))])

# Optional: bold headers and color significant rows
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4f81bd')
    elif table_data[row-1][5] == "Yes":
        cell.set_facecolor('#dff0d8')  # light green
    else:
        cell.set_facecolor('white')

plt.tight_layout()
plt.show()
