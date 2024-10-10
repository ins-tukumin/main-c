import pandas as pd

# CSVファイルを読み込みます（ファイル名を適宜変更してください）
input_file = "/mainfiles/dialy_file.csv"
df = pd.read_csv(input_file)

# user_id が7回登場しないデータを確認し、除外する
user_counts = df['user_id'].value_counts()
invalid_users = user_counts[user_counts != 7].index
print("7回登場しないuser_id: ", invalid_users.tolist())

# 7回登場しないuser_idの行を除外する
df = df[~df['user_id'].isin(invalid_users)]

# 日付をdatetime形式に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

df['StartDate'] = df['StartDate'].dt.date

# 比較用の日付（datetime.date型）を定義
start_date_0928 = pd.to_datetime('2024-09-28').date()
end_date_1001 = pd.to_datetime('2024-10-01').date()
date_1002 = pd.to_datetime('2024-10-02').date()
date_1003 = pd.to_datetime('2024-10-03').date()
date_1004 = pd.to_datetime('2024-10-04').date()

# 各ユーザーごとに必要な情報を抽出
output_data = []
for user_id, group in df.groupby('user_id'):
    # 日付順に並び替え
    group = group.sort_values('StartDate')
    
    # 09-28から10-01の日付範囲のQ1列の文字数カウント平均
    before_days = group[(group['StartDate'] >= start_date_0928) & 
                        (group['StartDate'] <= end_date_1001)]
    before_mean = round(before_days['Q1'].apply(len).mean(), 1)

    after_days = group[(group['StartDate'] >= date_1002) & 
                        (group['StartDate'] <= date_1004)]
    after_mean = round(after_days['Q1'].apply(len).mean(), 1)
    
    # 各日付ごとに文字数カウント（存在しない場合はデフォルト値 0 を使用）
    day1 = group[group['StartDate'] == date_1002]['Q1'].apply(len).values[0] if not group[group['StartDate'] == date_1002].empty else 0
    day2 = group[group['StartDate'] == date_1003]['Q1'].apply(len).values[0] if not group[group['StartDate'] == date_1003].empty else 0
    day3 = group[group['StartDate'] == date_1004]['Q1'].apply(len).values[0] if not group[group['StartDate'] == date_1004].empty else 0

    # 出力用データに追加
    output_data.append([user_id, before_mean, after_mean, day1, day2, day3])
# データフレームに変換
output_df = pd.DataFrame(output_data, columns=['user_id', 'before', 'after', 'day1', 'day2', 'day3'])

# CSVに保存
output_file = "mainfiles/mainfiles/dialy_count_temp.csv"
output_df.to_csv(output_file, index=False)

print(f"処理が完了しました。結果は {output_file} に保存されました。")