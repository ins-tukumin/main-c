import pandas as pd

# CSVファイルを読み込み（例: 'input.csv'）
df = pd.read_csv('mainfiles/analysis/1002_selected_renamed.csv')

# 抽出したい列のリストを指定（例: 'user_id', 'Q2', 'Q3' 列を抽出）
selected_columns = ['understanding']
# PANAS_P,PANAS_N,competence,warmth,satisfaction,effectiveness,efficiency,willingness,understanding

# 条件を設定：group 列が "A" の行のみを対象
df_filtered_a = df[df['group'] == 'groupa']
df_filtered_b = df[df['group'] == 'groupb']
df_filtered_c = df[df['group'] == 'groupc']

# 条件に合致した行から指定した列のみを抽出
df_selected_a = df_filtered_a[selected_columns]
df_selected_b = df_filtered_b[selected_columns]
df_selected_c = df_filtered_c[selected_columns]

# 新しいCSVファイルとして保存
df_selected_a.to_csv('mainfiles/analysis/filtered_output_a.csv', index=False)
df_selected_b.to_csv('mainfiles/analysis/filtered_output_b.csv', index=False)
df_selected_c.to_csv('mainfiles/analysis/filtered_output_c.csv', index=False)

print("指定した条件と列を含む 'filtered_output.csv' を作成しました。")