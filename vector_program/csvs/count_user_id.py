import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('mainfiles/1001.csv')

# StartDate列をdatetime型に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

# StartDate列から月日を日本語形式で抽出して新しい列を作成
df['StartDate_month_day'] = df['StartDate'].dt.strftime('%m月%d日')

# 2024-09-28 12:05:00以降のデータをフィルタリング
df = df[df['StartDate'] >= pd.Timestamp("2024-09-28 12:05:00")]

# user_idごとにStartDate_month_day列の値を羅列
user_id_dates = df.groupby('user_id')['StartDate_month_day'].apply(lambda x: ' '.join(x))

# 出現した各user_idとその羅列した日付を.txtに書き出す
with open('user_id_dates.txt', 'w') as f:
    for user_id, dates in user_id_dates.items():
        f.write(f'{user_id},{dates}\n')

print("user_idごとの日付を 'user_id_dates.txt' に書き出しました。")
