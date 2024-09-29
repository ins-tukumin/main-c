import os
import pandas as pd
import numpy as np

# CSVファイルの読み込み
file_path = 'mainfiles/0929test.csv'
df = pd.read_csv(file_path)

# 抜き出したい列名をリストで指定 'Q1' or 'Q34'
desired_columns = ['StartDate', 'Q1', 'user_id']
#desired_columns = ['StartDate', 'Q1', 'user_id', 'group']

# StartDate列をdatetime型に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

# 2024-09-28 12:05:00以降のデータをフィルタリング
df = df[df['StartDate'] >= pd.Timestamp("2024-09-28 12:05:00")]

# 指定した列だけを抜き出す（コピーではなく.locを使って操作）
df_selected = df.loc[:, desired_columns]

# StartDate列を抜き出し、日時型に変換して月日だけを取得
df_selected.loc[:, 'StartDate_month_day'] = pd.to_datetime(df_selected['StartDate']).dt.strftime('%m月%d日')

# 改行を削除（.locを使用して値を変更）
df_selected.loc[:, 'Q1'] = df_selected['Q1'].str.replace('\n', ' ').str.replace('\r', ' ')

# 欠損データを含む行を削除
df_selected = df_selected.dropna(subset=['user_id'])

# user_idに数値や文字列が混在することを想定し、すべて文字列に変換
df_selected['user_id'] = df_selected['user_id'].fillna('').astype(str)
#df_selected['user_id'] = df_selected['user_id'].astype(str)

# ユニークなuser_idを取得
unique_user_ids = df_selected['user_id'].unique()

# ユーザー数とグループ数を確認
n_users = len(unique_user_ids)
n_groups = 3

# グループのリスト
groups = ['groupa', 'groupb', 'groupc']

# ユーザーIDをランダムにシャッフル
np.random.shuffle(unique_user_ids)

# ユーザーIDをできるだけ均等に3つのグループに割り当て
group_assignment = np.array_split(unique_user_ids, n_groups)

# グループ割り当て結果をリストにまとめる
user_group_mapping = []
for group, users in zip(groups, group_assignment):
    for user_id in users:
        user_group_mapping.append({'user_id': user_id, 'group': group})

# user_group_mappingをデータフレームに変換
user_group_mapping_df = pd.DataFrame(user_group_mapping)

# グループごとの人数をカウントして表示
group_counts = user_group_mapping_df['group'].value_counts()
print("各グループの人数:")
print(group_counts)

# 元のデータフレームにuser_idに基づいてグループ情報を結合する
df_selected = pd.merge(df_selected, user_group_mapping_df, on='user_id', how='left')

# 入力ファイル名を取得（拡張子を除く）
input_file_name = os.path.splitext(os.path.basename(file_path))[0]

# 出力ファイル名を作成
output_file_path = f"mainfiles/{input_file_name}_selected_file.csv"

# 処理後のデータを新しいCSVファイルとして保存
df_selected.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# ユーザーIDとグループのペアを新しいデータフレームに抽出
df_group = user_group_mapping_df[['user_id', 'group']]

# group.txtとして保存する
group_file_path = '../../group_test.txt'
df_group.to_csv(group_file_path, index=False, header=False, sep=',', encoding='utf-8-sig')

# 出力ファイルのパスを表示
print(f"処理後のファイルは {output_file_path} に保存されました。")
print(f"グループ分けのファイルは {group_file_path} に保存されました。")
