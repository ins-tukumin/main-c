import pandas as pd

# CSVファイルを読み込む（ファイル名を適宜変更）
df1 = pd.read_csv('mainfiles/mainfiles/1004_selected_file.csv')  # 一つ目のCSV（df1）
df2 = pd.read_csv('mainfiles/mainfiles/1001_selected_file.csv')  # 二つ目のCSV（df2）

df1 = df1[df1['StartDate'] >= pd.Timestamp("2024-11-20 12:05:00")]
df2 = df2[df2['StartDate'] >= pd.Timestamp("2024-11-20 12:05:00")]

# 欠損データを含む行を削除
df1 = df1.dropna(subset=['user_id'])
df2 = df2.dropna(subset=['user_id'])

# user_idに数値や文字列が混在することを想定し、すべて文字列に変換
df1['user_id'] = df1['user_id'].fillna('').astype(str)
df2['user_id'] = df2['user_id'].fillna('').astype(str)

# df2のQ34列をQ1にリネーム
df1.rename(columns={'Q34': 'Q1'}, inplace=True)

# 2つのDataFrameを縦方向に結合（行を追加する場合）
combined_df = pd.concat([df1, df2], ignore_index=True)

# user_id の登場回数をカウント
user_id_counts = combined_df['user_id'].value_counts()

# 7回以外のuser_idを抽出
non_seven_user_ids = user_id_counts[user_id_counts != 7]

# 該当する user_id を表示
if not non_seven_user_ids.empty:
    print("7回以外登場したuser_id:")
    print(non_seven_user_ids)
else:
    print("すべてのuser_idは7回登場しています。")

# 結合した結果を新しいCSVとして保存
combined_df.to_csv("mainfiles/mainfiles/dialy_file.csv", index=False)

print("2つのCSVが1つに結合されました！")
