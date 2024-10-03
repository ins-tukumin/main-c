import pandas as pd

# CSVファイルを読み込み（ファイル名を適宜変更）
df1 = pd.read_csv('mainfiles/analysis/1002_selected_renamed.csv')  # 一つ目のCSV（df1）
df2 = pd.read_csv('mainfiles/0928.csv')  # 二つ目のCSV（df2）

# df1 のすべての列を保持したまま、df2 の Q2, Q3 列を user_id でマージ
df1 = df1.merge(df2[['user_id', 'Q2', 'Q3']], on='user_id', how='left')

df1.rename(columns={'Q2': 'jender'}, inplace=True)
df1.rename(columns={'Q3': 'age'}, inplace=True)

# 結果を表示
print("結合後のデータフレーム：")
print(df1)

# 必要なら結合結果をCSVとして保存
df1.to_csv('mainfiles/analysis/merged_output.csv', index=False)