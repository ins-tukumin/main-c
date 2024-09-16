import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('taste.csv')

# user_idの出現回数をカウント
user_id_counts = df['user_id'].value_counts()

# 出現回数を.txtに書き出す
with open('user_id_counts.txt', 'w') as f:
    for user_id, count in user_id_counts.items():
        f.write(f'{user_id}: {count}\n')

print("user_idの出現回数を 'user_id_counts.txt' に書き出しました。")
