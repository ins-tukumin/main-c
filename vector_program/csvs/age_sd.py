import pandas as pd

# CSVファイルを読み込む
# 例: "data.csv" が読み込むファイルの名前
data = pd.read_csv("mainfiles/analysis/1002_age_gender.csv")

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