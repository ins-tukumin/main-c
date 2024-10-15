import pandas as pd

# 2つのCSVファイルを読み込む
df1 = pd.read_csv('mainfiles/analysis/transformed_data.csv')
df2 = pd.read_csv('mainfiles/analysis/cos_for_analysis.csv')

# 'user_id'列でマージ（inner join: 共通するuser_idのみ結合）
merged_df = pd.merge(df1, df2, on='user_id', how='inner')

# 結果を確認
print(merged_df.head())

# 新しいCSVファイルとして保存
merged_df.to_csv('merged_data.csv', index=False)
