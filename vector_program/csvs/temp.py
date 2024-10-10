import pandas as pd

# CSVファイルを読み込む
data1 = pd.read_csv("mainfiles/analysis/dialy_file.csv")  # 最初のCSVファイル
data2 = pd.read_csv("mainfiles/0928.csv")  # 2つ目のCSVファイル

# user_id 列の値を取得
user_ids_1 = set(data1['user_id'])  # file1のuser_id
user_ids_2 = set(data2['user_id'])  # file2のuser_id

# どちらか一方にのみ存在する user_id を取得
only_in_file1 = user_ids_1 - user_ids_2  # file1 にのみ存在する user_id
only_in_file2 = user_ids_2 - user_ids_1  # file2 にのみ存在する user_id

# どちらか一方にのみ存在する user_id を表示
print("User IDs only in file1:")
print(only_in_file1)

print("\nUser IDs only in file2:")
print(only_in_file2)

# 両方に存在する user_id を取得し、その数をカウント
common_user_ids = user_ids_1 & user_ids_2
print(f"\nNumber of common user IDs in both files: {len(common_user_ids)}")