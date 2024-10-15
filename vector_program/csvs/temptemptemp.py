import pandas as pd

# CSVファイルを読み込む
file_path = 'conversation_log_analysis_groupc_cos_diary.csv'  # あなたのCSVファイルのパスに置き換えてください
df = pd.read_csv(file_path, encoding='shift_jis')

# データフレームをmeltして、指定された列名を1列にまとめる
value_vars = [
    'day1_AI', 'day1_Human', 'day1_cos_AI_Human', 'day1_cos_diary_AI', 'day1_cos_diary_Human',
    'day2_AI', 'day2_Human', 'day2_cos_AI_Human', 'day2_cos_diary_AI', 'day2_cos_diary_Human',
    'day3_AI', 'day3_Human', 'day3_cos_AI_Human', 'day3_cos_diary_AI', 'day3_cos_diary_Human'
]

df_melted = pd.melt(df, id_vars=['userid'], 
                    value_vars=value_vars,
                    var_name='Day_Type', value_name='Score')

# 'Day_Type' を 'Day' と 'Type' に分割する ('_' を基に分割)
df_melted[['Day', 'Type']] = df_melted['Day_Type'].str.extract(r'(day\d+)_(.*)')

# ピボットテーブルで 'AI', 'Human', 'cos_AI_Human', 'cos_diary_AI', 'cos_diary_Human' の列を作成
df_pivot = df_melted.pivot_table(index=['userid', 'Day'], columns='Type', values='Score').reset_index()

# 列名を変更してわかりやすくする
df_pivot.columns.name = None  # マルチインデックス名を削除
df_pivot = df_pivot.rename(columns={
    'userid': 'user_id', 
    'AI': 'AI', 
    'Human': 'Human',
    'cos_AI_Human': 'cos_AI_Human', 
    'cos_diary_AI': 'cos_diary_AI',
    'cos_diary_Human': 'cos_diary_Human'
})

# すべての数値データを小数点以下2桁に丸める
df_pivot = df_pivot.round(3)

# 出力ファイルに保存
output_file_path = 'final_groupc.csv'  # 出力ファイルパスに置き換え
df_pivot.to_csv(output_file_path, index=False)

# 結果を表示
df_pivot.head()
