import pandas as pd

# CSVファイルを読み込む
csv_file_1 = "mainfiles/1001_selected_file.csv"
csv_file_2 = "mainfiles/1002_selected_file.csv"

# dtype={'user_id': str} を指定して user_id を文字列型に強制
df1 = pd.read_csv(csv_file_1, dtype={'user_id': str})
df2 = pd.read_csv(csv_file_2, dtype={'user_id': str})

# df2のQ34列をQ1にリネーム
df2.rename(columns={'Q34': 'Q1'}, inplace=True)

# 2つのDataFrameを縦方向に結合（行を追加する場合）
combined_df = pd.concat([df1, df2], ignore_index=True)

# 結合した結果を新しいCSVとして保存
combined_df.to_csv("mainfiles/1002_combined_file.csv", index=False)

print("2つのCSVが1つに結合されました！")
