import pandas as pd
from sklearn.preprocessing import StandardScaler

# CSVファイルを読み込む
file_path = 'mainfiles/analysis/conversation_log_analysis_gourpb.csv'  # あなたのCSVファイルのパスに置き換えてください
df = pd.read_csv(file_path, encoding='shift_jis')

# データフレームをmeltして、day1_AI, day2_AI, day3_AI, day1_Human, day2_Human, day3_Humanを1列にまとめる
df_melted = pd.melt(df, id_vars=['userid'], 
                    value_vars=['day1_AI', 'day2_AI', 'day3_AI', 'day1_Human', 'day2_Human', 'day3_Human'],
                    var_name='Day_Type', value_name='Score')

# 'Day_Type' を 'Day' と 'Type' に分割する
df_melted[['Day', 'Type']] = df_melted['Day_Type'].str.split('_', expand=True)

# 'AI' と 'Human' の列を作るためにピボットテーブルを使用
df_pivot = df_melted.pivot_table(index=['userid', 'Day'], columns='Type', values='Score').reset_index()

# 列の名前を変更してわかりやすくする
df_pivot.columns.name = None  # マルチインデックス名を削除
df_pivot = df_pivot.rename(columns={'userid': 'user_id', 'AI': 'AI', 'Human': 'Human'})

# Zスコア標準化をユーザーごとに実行
scaler = StandardScaler()
df_grouped = df_pivot.groupby('user_id')

# 各ユーザーのスコアを標準化する
df_pivot['AI_std'] = df_grouped['AI'].transform(lambda x: (x - x.mean()) / x.std())
df_pivot['Human_std'] = df_grouped['Human'].transform(lambda x: (x - x.mean()) / x.std())

# 標準化された結果を保存
output_file_path = 'mainfiles/analysis/groupb_log_csv_scaled.csv'  # 出力ファイルパスに置き換え
df_pivot.to_csv(output_file_path, index=False)

# 結果を表示
df_pivot.head()
