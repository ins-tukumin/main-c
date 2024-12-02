import pandas as pd

# CSVファイルのパス
csv_path = "mainfiles/analysis/dialy_file_added_group.csv"  # 対象のCSVファイル

# CSVを読み込む
df = pd.read_csv(csv_path)

# user_idごとの出現回数を集計
user_counts = df['user_id'].value_counts()

# 出現回数が3回以下のuser_idを取得
filtered_user_ids = user_counts[user_counts <= 5].index.tolist()

# 出現回数が3回以下のuser_idをprint
print("User IDs appearing 3 times or fewer:")
for user_id in filtered_user_ids:
    print(f"user_id: {user_id}, count: {user_counts[user_id]}")
