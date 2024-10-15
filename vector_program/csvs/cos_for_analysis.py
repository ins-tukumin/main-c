import pandas as pd

# CSVファイルの読み込み
group_a = pd.read_csv('mainfiles/analysis/groupa_output.csv')
group_b = pd.read_csv('mainfiles/analysis/groupb_output.csv')
group_c = pd.read_csv('mainfiles/analysis/groupc_output.csv')

# グループ名の列を追加
group_a['group'] = 'groupa'
group_b['group'] = 'groupb'
group_c['group'] = 'groupc'

# 縦につなげる
combined_data = pd.concat([group_a, group_b, group_c], ignore_index=True)

# 数値データを小数点以下3桁に揃える
combined_data = combined_data.round(3)

# 結果をCSVに保存
combined_data.to_csv('hyoutei.csv', index=False)

# 出力結果を表示
print(combined_data)
