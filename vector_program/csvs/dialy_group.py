import pandas as pd

# 1. CSVファイルの読み込み
# csv1 = pd.read_csv('../../group_assignment.csv')  # 一つ目のCSV (user_id, group列が含まれる)
# csv2 = pd.read_csv('mainfiles/mainfiles/dialy_count.csv')  # 二つ目のCSV (user_idを含む様々な列が存在する)

# 2. user_idを基に二つのデータフレームをマージする
# merged_df = pd.merge(csv2, csv1[['user_id', 'group']], on='user_id', how='left')

# 1. データを読み込む（例: data.csv）
data = pd.read_csv('mainfiles/analysis/dialy_file.csv')

# 2. 各グループにデータを分ける
group_a = data[data['group'] == 'groupa'][['user_id', 'before', 'after', 'day1', 'day2', 'day3']]
group_b = data[data['group'] == 'groupb'][['user_id', 'before', 'after', 'day1', 'day2', 'day3']]
group_c = data[data['group'] == 'groupc'][['user_id', 'before', 'after', 'day1', 'day2', 'day3']]

# 3. 各グループごとにファイルを保存する
group_a.to_csv("mainfiles/analysis/groupa_data.csv", index=False)
group_b.to_csv("mainfiles/analysis/groupb_data.csv", index=False)
group_c.to_csv("mainfiles/analysis/groupc_data.csv", index=False)

# 4. 保存したファイル名と行数を表示
print("groupa_data.csv の行数:", len(group_a))
print("groupb_data.csv の行数:", len(group_b))
print("groupc_data.csv の行数:", len(group_c))

# 3. 新しいファイルとして出力する
# merged_df.to_csv('mainfiles/mainfiles/dialy_group.csv', index=False)

# 4. 一つ目のCSVに存在するのに二つ目のCSVには存在しなかったuser_idを抽出して表示
# missing_users = set(csv1['user_id']) - set(csv2['user_id'])
# print("Missing user_ids in the second CSV:", missing_users)