import pandas as pd

# CSVファイルのパス
csv1_path = "BIGBERT_old.csv"  # 1つ目のCSV（ベースとなるCSV）
csv2_path = "AUTO_protocol_c.csv"  # 2つ目のCSV
csv3_path = "AUTO_protocol_rep.csv"  # 3つ目のCSV

# 取得したい列をリストで指定（'user_id' は必須）
columns_csv2 = ["user_id","total_topic","stan_topic_count","stan_none_count"]  # 2つ目のCSVから取得したい列
columns_csv3 = ["user_id","total_topic","stan_topic_count","stan_none_count"]  # 3つ目のCSVから取得したい列

# CSVを読み込む
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path, usecols=columns_csv2)  # 指定した列だけ読み込む
df3 = pd.read_csv(csv3_path, usecols=columns_csv3)  # 指定した列だけ読み込む

# user_id を文字列型に変換
df1['user_id'] = df1['user_id'].astype(str)
df2['user_id'] = df2['user_id'].astype(str)
df3['user_id'] = df3['user_id'].astype(str)

# user_id を基準に結合（左結合）
merged_df = df1.merge(df2, on="user_id", how="left")  # 2つ目のCSVと結合
merged_df = merged_df.merge(df3, on="user_id", how="left")  # 3つ目のCSVと結合

# 衝突した列を1つに統合
merged_df["total_topic"] = merged_df["total_topic_x"].combine_first(merged_df["total_topic_y"])
merged_df["stan_topic_count"] = merged_df["stan_topic_count_x"].combine_first(merged_df["stan_topic_count_y"])
merged_df["stan_none_count"] = merged_df["stan_none_count_x"].combine_first(merged_df["stan_none_count_y"])

# 不要な列（_x, _y）を削除
merged_df = merged_df.drop(columns=["total_topic_x", "total_topic_y", "stan_topic_count_x", "stan_topic_count_y", "stan_none_count_x", "stan_none_count_y"])

# 結果を確認
# print(merged_df)

# 必要に応じて保存
output_path = "BIGBERT.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged CSV saved to {output_path}")
