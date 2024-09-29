import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('mainfiles/0928test2.csv')

# StartDate列をdatetime型に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

# 2024-09-28 12:05:00以降のデータをフィルタリング
df = df[df['StartDate'] >= pd.Timestamp("2024-09-28 12:05:00")]

# user_idの出現回数をカウント
user_id_counts = df['user_id'].value_counts()

# 出現回数を.txtに書き出す
with open('user_id_counts.txt', 'w') as f:
    for user_id, count in user_id_counts.items():
        f.write(f'{user_id}: {count}\n')

print("user_idの出現回数を 'user_id_counts.txt' に書き出しました。")
