import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv("data/processed/marketing_AB_clean.csv")

conv = df.groupby("test_group")["converted"].agg(["sum", "count"])
n_ad, n_psa = conv.loc["ad", "count"], conv.loc["psa", "count"]
c_ad, c_psa = conv.loc["ad", "sum"], conv.loc["psa", "sum"]

p_ad, p_psa = c_ad / n_ad, c_psa / n_psa
print(f"ad 组: {c_ad}/{n_ad} = {p_ad:.6f}")
print(f"psa 组: {c_psa}/{n_psa} = {p_psa:.6f}")

# 单侧两比例 z 检验：H0: p_ad <= p_psa, H1: p_ad > p_psa
count = np.array([c_ad, c_psa])
nobs = np.array([n_ad, n_psa])
z_stat, p_val = proportions_ztest(count, nobs, alternative="larger")
print(f"\n单侧 z 检验: z={z_stat:.4f}, p={p_val:.6e}")

# 差值 95% 置信区间（手动 Wald 法，避免版本兼容问题）
diff = p_ad - p_psa
se = np.sqrt(p_ad * (1 - p_ad) / n_ad + p_psa * (1 - p_psa) / n_psa)
ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
print(f"转化率差（ad - psa）: {diff:.6f}")
print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

# 效应量
abs_lift = diff
rel_lift = diff / p_psa
print(f"\n绝对提升: {abs_lift*100:.4f} pp")
print(f"相对提升: {rel_lift*100:.2f}%")