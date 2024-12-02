import pandas as pd

# CSVファイルのパス
csv1_path = "mainfiles/analysis/dialy_file.csv"  # 1つ目のCSVファイル（間違ったgroup列）
csv2_path = "../../group_assignment.csv"  # 2つ目のCSVファイル（正しいgroup列）

# CSVを読み込む
df1 = pd.read_csv(csv1_path)  # 1つ目のCSV
df2 = pd.read_csv(csv2_path)  # 2つ目のCSV

# `user_id` を基準にgroup列を上書き
updated_groups = 0
for _, row in df2.iterrows():
    user_id = row['user_id']
    group_value = row['group']

    # user_idが1つ目のCSVに存在するか確認
    if user_id in df1['user_id'].values:
        # すべての該当行を上書き
        affected_rows = df1.loc[df1['user_id'] == user_id, 'group']
        df1.loc[df1['user_id'] == user_id, 'group'] = group_value
        updated_groups += len(affected_rows)
    else:
        # 2つ目のCSVにあるが1つ目にはない場合
        print(f"user_id {user_id} is in the second CSV but not in the first CSV.")

# 結果の確認
print(f"Updated group values for {updated_groups} rows.")

# 上書き後のCSVを保存
output_path = "mainfiles/analysis/dialy_file_added_group.csv"
df1.to_csv(output_path, index=False, encoding="utf-8")
print(f"Updated CSV saved to {output_path}.")
