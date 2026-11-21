import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/processed/marketing_AB_clean.csv")

# ================= 按星期 =================
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_stats = df.groupby(["most_ads_day", "test_group"])["converted"].agg(["count", "mean"]).reset_index()
day_pivot = day_stats.pivot(index="most_ads_day", columns="test_group", values="mean").reindex(day_order)
day_n = day_stats.pivot(index="most_ads_day", columns="test_group", values="count").reindex(day_order)
day_pivot["diff_pp"] = (day_pivot["ad"] - day_pivot["psa"]) * 100
print("--- 按星期：两组转化率与差值 ---")
print(day_pivot.round(6))
day_pivot.to_csv("outputs/tables/day_conversion.csv")

# ================= 按小时（带过滤） =================
hour_stats = df.groupby(["most_ads_hour", "test_group"])["converted"].agg(["count", "sum", "mean"]).reset_index()
hour_pivot = hour_stats.pivot(index="most_ads_hour", columns="test_group", values="mean")
hour_n = hour_stats.pivot(index="most_ads_hour", columns="test_group", values="count")
hour_c = hour_stats.pivot(index="most_ads_hour", columns="test_group", values="sum")

hour_pivot["diff_pp"] = (hour_pivot["ad"] - hour_pivot["psa"]) * 100
hour_pivot["psa_n"] = hour_n["psa"]
hour_pivot["psa_conv"] = hour_c["psa"]

# 过滤：psa 样本 >= 500 且转化数 >= 5
valid_mask = (hour_pivot["psa_n"] >= 500) & (hour_pivot["psa_conv"] >= 5)
hour_valid = hour_pivot[valid_mask].copy()
hour_invalid = hour_pivot[~valid_mask].copy()

print("\n--- 按小时：纳入分析的时段（psa样本>=500且转化数>=5） ---")
print(hour_valid[["ad", "psa", "diff_pp", "psa_n", "psa_conv"]].round(6))

print("\n--- 按小时：剔除的时段 ---")
print(hour_invalid[["ad", "psa", "diff_pp", "psa_n", "psa_conv"]].round(6))

hour_pivot.to_csv("outputs/tables/hour_conversion_all.csv")
hour_valid.to_csv("outputs/tables/hour_conversion_valid.csv")
print("\n已保存: hour_conversion_all.csv, hour_conversion_valid.csv")

# ================= 热力图（只保留有效小时） =================
heat = df[df["test_group"] == "ad"].groupby(["most_ads_day", "most_ads_hour"])["converted"].mean().unstack()
heat = heat.reindex(day_order)
heat = heat[valid_mask[valid_mask].index]  # 只保留有效小时列

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(7))
ax.set_yticklabels(day_order)
ax.set_xlabel("most_ads_hour（仅保留psa样本充足的时段）")
ax.set_ylabel("most_ads_day")
ax.set_title("ad 组转化率：星期 × 有效小时")
plt.colorbar(im, ax=ax, label="转化率")
plt.tight_layout()
plt.savefig("outputs/figures/heatmap_day_hour_ad.png", dpi=150)
print("\n已保存: outputs/figures/heatmap_day_hour_ad.png")