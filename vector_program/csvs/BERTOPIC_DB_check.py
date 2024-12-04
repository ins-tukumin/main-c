import pandas as pd

# CSVファイルの読み込み
csv1_path = 'mainfiles/analysis/dialy_file.csv'  # 1つ目のCSVファイルのパス
csv2_path = 'BIGBERT.csv'  # 2つ目のCSVファイルのパス
output_path = 'mainfiles/analysis/FINAL_dialy_file.csv'  # 出力先のCSVファイルのパス

# データフレームを読み込む
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# 両方のCSVに存在するuser_idを取得
common_user_ids = set(df1['user_id']).intersection(set(df2['user_id']))

# 削除するuser_idを特定
removed_user_ids = set(df1['user_id']) - common_user_ids

# 削除されたuser_idを出力
print("削除されたuser_id:")
print(removed_user_ids)

# 1つ目のCSVをフィルタリング
filtered_df1 = df1[df1['user_id'].isin(common_user_ids)]

# 結果を新しいCSVに保存
filtered_df1.to_csv(output_path, index=False)

print(f"フィルタリング後のCSVを保存しました: {output_path}")
