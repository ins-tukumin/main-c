import numpy as np
import pandas as pd

# CSVファイルの読み込み
file_path = 'mainfiles/1001.csv'
df = pd.read_csv(file_path)

# 優先的に groupa に割り当てたいユーザーIDのリスト
priority_users = ['878721', '94338', '95866','acu9751cw','akimi381','khi1_b4aa','KoHaRu09','makimaki0519','mor2kir1500','naotok24','quartet456','rdkny933','SGヘルスケア','shigenoi','ys15_cw','Yuko_office','yume43','にゃすも']

# StartDate列をdatetime型に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

# 2024-09-28 12:05:00以降のデータをフィルタリング
df = df[df['StartDate'] >= pd.Timestamp("2024-09-28 12:05:00")]

# ユニークなuser_idを取得
unique_user_ids = df['user_id'].unique()

# 1. ユーザー数とグループ数を設定
n_users = len(unique_user_ids)
n_groups = 3

# 2. グループのリスト
groups = ['groupa', 'groupb', 'groupc']

# 3. 優先ユーザーを groupa に割り当て、残りのユーザーリストを更新
group_a_users = [user for user in priority_users if user in unique_user_ids]
remaining_users = [user for user in unique_user_ids if user not in group_a_users]

# 4. 残りのユーザーをランダムにシャッフル
np.random.shuffle(remaining_users)

# 5. 優先ユーザーを groupa に追加した後、残りのユーザーをすべてのグループに均等に割り当て
#    初期割り当て用の辞書を定義
group_assignment = {'groupa': group_a_users, 'groupb': [], 'groupc': []}

# 6. 各グループのターゲットサイズを算出
target_size = n_users // n_groups

# 7. 残りのユーザーを各グループに順番に均等割り当て
for user in remaining_users:
    # 各グループのサイズを確認し、均等になるよう割り当て
    min_group = min(group_assignment, key=lambda k: len(group_assignment[k]))
    group_assignment[min_group].append(user)

# 8. グループ割り当て結果をリストにまとめる
user_group_mapping = []
for group, users in group_assignment.items():
    for user_id in users:
        user_group_mapping.append({'user_id': user_id, 'group': group})

# 結果をデータフレームに変換
user_group_mapping_df = pd.DataFrame(user_group_mapping)

# 各グループの人数を確認
group_counts = user_group_mapping_df['group'].value_counts()
print("各グループの人数:")
print(group_counts)

# 出力結果
print(user_group_mapping_df)

# ユーザーIDとグループのペアを新しいデータフレームに抽出
df_group = user_group_mapping_df[['user_id', 'group']]

# group.txtとして保存する
group_file_path = '../../group_test.txt'
df_group.to_csv(group_file_path, index=False, header=False, sep=',', encoding='utf-8-sig')
