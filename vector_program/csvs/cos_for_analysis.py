import pandas as pd

# CSVファイルの読み込み
group_a = pd.read_csv('BERT_cos_a.csv', encoding='shift_jis')
group_b = pd.read_csv('BERT_cos_b.csv', encoding='shift_jis')
group_c = pd.read_csv('BERT_cos_c.csv', encoding='shift_jis')

# グループ名の列を追加
group_a['group'] = 'c'
group_b['group'] = 'r'
group_c['group'] = 'rep'

# 縦につなげる
combined_data = pd.concat([group_a, group_b, group_c], ignore_index=True)

# 数値データを小数点以下3桁に揃える
# combined_data = combined_data.round(3)

# 結果をCSVに保存
combined_data.to_csv('BERT.csv', index=False)

# 出力結果を表示
# print(combined_data)
