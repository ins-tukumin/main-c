import os
import pandas as pd
import numpy as np

# CSVファイルの読み込み
file_path = 'mainfiles/1002.csv'
df = pd.read_csv(file_path)

# 抜き出したい列名をリストで指定 'Q1' or 'Q34'
desired_columns = ['StartDate', 'user_id','Q2_1','Q2_2','Q2_3','Q2_4','Q2_5','Q2_6','Q2_7','Q2_8','Q2_9','Q2_10','Q2_11','Q2_12','Q2_13','Q2_14','Q2_15','Q2_16','Q6','Q8','Q10','Q12','Q14','Q16','Q20','Q22','Q24','Q28','Q30']
#desired_columns = ['StartDate', 'Q1', 'user_id', 'group']

# user_id を文字列型に変換（数値と文字列が混在することを想定）
df['user_id'] = df['user_id'].astype(str)

# StartDate列をdatetime型に変換
df['StartDate'] = pd.to_datetime(df['StartDate'])

# 2024-09-28 12:05:00以降のデータをフィルタリング
df = df[df['StartDate'] >= pd.Timestamp("2024-10-02 12:05:00")]

# 指定した列だけを抜き出す（コピーではなく.locを使って操作）
df_selected = df.loc[:, desired_columns]

# StartDate列を抜き出し、日時型に変換して月日だけを取得
df_selected.loc[:, 'StartDate_month_day'] = pd.to_datetime(df_selected['StartDate']).dt.strftime('%m月%d日')

# 改行を削除（.locを使用して値を変更）
#df_selected.loc[:, 'Q1'] = df_selected['Q1'].str.replace('\n', ' ').str.replace('\r', ' ')

# 欠損データを含む行を削除
df_selected = df_selected.dropna(subset=['user_id'])

# user_idに数値や文字列が混在することを想定し、すべて文字列に変換
df_selected['user_id'] = df_selected['user_id'].fillna('').astype(str)

df_selected['PANAS_P'] = df[['Q2_1','Q2_2','Q2_3','Q2_4','Q2_5','Q2_6','Q2_7','Q2_8']].mean(axis=1).round(3)
df_selected['PANAS_N'] = df[['Q2_9','Q2_10','Q2_11','Q2_12','Q2_13','Q2_14','Q2_15','Q2_16']].mean(axis=1).round(3)

df_selected['competence'] = df[['Q6','Q8','Q10']].mean(axis=1).round(3)
df_selected['warmth'] = df[['Q12','Q14','Q16']].mean(axis=1).round(3)

# 列名を変更 ('旧列名'を'新列名'に変更)
df.rename(columns={'旧列名': '新列名'}, inplace=True)

# 列名を変更 ('旧列名'を'新列名'に変更)
df.rename(columns={'旧列名': '新列名'}, inplace=True)