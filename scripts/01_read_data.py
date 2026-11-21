import pandas as pd

# ---------- 1. 读原始数据 ----------
raw_path = "C:/Users/wuhan/ad_traffic_ab_test/data/raw/marketing_AB.csv.zip"
df = pd.read_csv(raw_path)
print("原始形状:", df.shape)
print("原始列名:", df.columns.tolist())

# ---------- 2. 清洗 ----------
# 去掉 CSV 自带的索引列
df = df.drop(columns=["Unnamed: 0"])
# 列名空格转下划线，后续引用更方便
df.columns = [c.replace(" ", "_") for c in df.columns]

# ---------- 3. 验证 ----------
print("\n清洗后形状:", df.shape)
print("清洗后列名:", df.columns.tolist())

print("\n--- 缺失值 ---")
print(df.isnull().sum())

print("\n--- test_group 样本量与转化率 ---")
print(df.groupby("test_group")["converted"].agg(["count", "sum", "mean"]))

print("\n--- most_ads_day 分布 ---")
print(df["most_ads_day"].value_counts())

print("\n--- most_ads_hour 范围 ---")
print("min:", df["most_ads_hour"].min(), "max:", df["most_ads_hour"].max())

print("\n--- total_ads 描述 ---")
print(df["total_ads"].describe())

# ---------- 4. 保存干净版 ----------
out_path = "C:/Users/wuhan/ad_traffic_ab_test/data/processed/marketing_AB_clean.csv"
df.to_csv(out_path, index=False)
print(f"\n已保存: {out_path}")