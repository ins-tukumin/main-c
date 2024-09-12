import os
import pandas as pd

# CSVファイルの読み込み
file_path = 'test.csv'
df = pd.read_csv(file_path)

# 抜き出したい列名をリストで指定
desired_columns = ['StartDate', 'Q1', 'user_id']
#desired_columns = ['StartDate', 'Q1', 'user_id', 'group']

# 指定した列だけを抜き出す
df_selected = df[desired_columns]

# StartDate列を抜き出し、日時型に変換して月日だけを取得
df_selected['StartDate_month_day'] = pd.to_datetime(df_selected['StartDate']).dt.strftime('%m月%d日')

# 改行を削除
df_selected['Q1'] = df_selected['Q1'].str.replace('\n', ' ').str.replace('\r', ' ')

# 欠損データを含む行を削除
df_selected = df_selected.dropna(subset=['user_id'])

# 文字列に変換
df_selected['user_id'] = df_selected['user_id'].astype(int).astype(str)

# 入力ファイル名を取得（拡張子を除く）
input_file_name = os.path.splitext(os.path.basename(file_path))[0]

# 出力ファイル名を作成
output_file_path = f"{input_file_name}_selected_file.csv"

# 処理後のデータを新しいCSVファイルとして保存
df_selected.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# 出力ファイルのパスを表示
print(f"処理後のファイルは {output_file_path} に保存されました。")
