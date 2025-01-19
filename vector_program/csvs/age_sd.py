import pandas as pd

# CSVファイルを読み込む
# 例: "data.csv" が読み込むファイルの名前
data = pd.read_csv("mainfiles/analysis/1002_age_gender.csv")
data2 = pd.read_csv("mainfiles/analysis/1002_selected_renamed.csv")
# group?を削除
data = data[data['group'] != 'groupb']

# gender 列の値をカウント
gender_counts = data['jender'].value_counts()
print("Gender Counts:")
print(gender_counts)

# age 列の平均、標準偏差、最大値、最小値を計算
age_mean = data['age'].mean()
age_sd = data['age'].std()
age_max = data['age'].max()
age_min = data['age'].min()

# 計算結果を表示
print("\nAge Statistics:")
print(f"Mean: {age_mean}")
print(f"Standard Deviation: {age_sd}")
print(f"Maximum: {age_max}")
print(f"Minimum: {age_min}")

# DataFrameとして読み込む
df1 = data
df2 = data2

# `user_id`の集合を取得
user_ids_1 = set(df1['user_id'])
user_ids_2 = set(df2['user_id'])

# 一方にしか存在しない`user_id`を取得
only_in_file1 = user_ids_1 - user_ids_2
only_in_file2 = user_ids_2 - user_ids_1

# 結果を出力
print("----------------------------------")
print("Only in file1:", only_in_file1)
print("Only in file2:", only_in_file2)