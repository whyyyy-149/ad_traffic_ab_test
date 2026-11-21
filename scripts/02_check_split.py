import numpy as np
import pandas as pd
from scipy.stats import chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

df = pd.read_csv("data/processed/marketing_AB_clean.csv")

counts = df["test_group"].value_counts()
n_ad, n_psa = counts["ad"], counts["psa"]
total = n_ad + n_psa
print(f"ad 组: {n_ad}, psa 组: {n_psa}, 合计: {total}")
print(f"实际比例: ad {n_ad/total:.2%}, psa {n_psa/total:.2%}")

# ---------- 任务1：分流比例情景分析 ----------
obs = np.array([n_ad, n_psa])
exp_a = np.array([total / 2, total / 2])
chi_a, p_a = chisquare(obs, f_exp=exp_a)
print(f"\n情景A（50/50）: chi2={chi_a:.2f}, p={p_a:.4e}")

exp_b = np.array([total * 0.96, total * 0.04])
chi_b, p_b = chisquare(obs, f_exp=exp_b)
print(f"情景B（96/4）: chi2={chi_b:.2f}, p={p_b:.4f}")

# ---------- 任务2：样本量敏感性分析 ----------
baseline = df.loc[df["test_group"] == "psa", "converted"].mean()
print(f"\n基线转化率(psa): {baseline:.6f}")
print(f"psa 实际样本量: {n_psa}")

power_analysis = NormalIndPower()
rows = []

for alternative in ["two-sided", "larger"]:
    for alpha in [0.05, 0.10]:
        for power in [0.75, 0.80]:
            for mde in [0.20, 0.30]:
                target = baseline * (1 + mde)
                effect = proportion_effectsize(target, baseline)
                n_req = power_analysis.solve_power(
                    effect_size=effect,
                    alpha=alpha,
                    power=power,
                    ratio=1,
                    alternative=alternative,
                )
                rows.append({
                    "检验类型": "单侧" if alternative == "larger" else "双侧",
                    "置信水平": f"{(1-alpha)*100:.0f}%",
                    "功效": power,
                    "MDE": f"相对+{mde*100:.0f}%",
                    "所需样本量": int(np.ceil(n_req)),
                    "psa实际": n_psa,
                    "是否充足": "是" if n_psa >= n_req else "否",
                })

result = pd.DataFrame(rows)
print("\n--- 样本量敏感性分析 ---")
print(result.to_string(index=False))

result.to_csv("outputs/tables/sample_size_sensitivity.csv", index=False)
print("\n已保存: outputs/tables/sample_size_sensitivity.csv")