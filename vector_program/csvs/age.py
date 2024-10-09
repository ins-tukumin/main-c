import pandas as pd

# CSVファイルを読み込む (ファイル名を指定してください)
df = pd.read_csv('mainfiles/analysis/1002_age_gender.csv')

# 年齢を3つのグループに分けて新しい列 'age_group' を作成
def assign_age_group(age):
    if 20 <= age <= 33:
        return 'young'
    elif 34 <= age <= 46:
        return 'middle'
    elif 47 <= age <= 60:
        return 'old'
    else:
        return 'out_of_range'  # もし20歳未満または60歳超の場合

# age_group 列を作成
df['age_group'] = df['age'].apply(assign_age_group)

# 結果を表示
print(df)

# 必要に応じて新しいCSVに保存
df.to_csv('mainfiles/analysis/1002_age_gender.csv', index=False)