import pandas as pd

# 1. CSVファイルの読み込み
day1 = pd.read_csv('mainfiles/analysis/1002_selected_renamed.csv')
day2 = pd.read_csv('mainfiles/analysis/1003_selected_renamed.csv')
day3 = pd.read_csv('mainfiles/analysis/1004_selected_renamed.csv')

# 2. 各データフレームに 'day' 列を追加
day1['day'] = 1
day2['day'] = 2
day3['day'] = 3

# 3. データを縦に結合（ロング型に変換）
merged_df = pd.concat([day1, day2, day3], axis=0)

# 4. `group` 列ごとにデータを分割し、各グループごとにCSV出力
for group_name in ['groupa', 'groupb', 'groupc']:
    # グループごとにデータを抽出
    group_df = merged_df[merged_df['group'] == group_name]
    
    # 各グループのデータをCSVファイルに出力
    output_file = f'mainfiles/analysis/{group_name}_output.csv'
    group_df.to_csv(output_file, index=False)
    print(f'Output file created: {output_file}')