import pandas as pd

# CSVファイルを読み込む
csv1 = pd.read_csv('BERT.csv')  # 1つ目のCSV
csv2 = pd.read_csv('BIGDATA.csv')  # 2つ目のCSV

# 1つ目のCSVから特定の列（例：'target_column'）をuser_idに基づいて抽出
# user_idと必要な列だけを保持
csv1_selected = csv1[['user_id', 'ave_cos_BERT_diary_Human']]

# 2つ目のCSVに1つ目のCSVから選んだ列をuser_idで結合（inner join）
merged_csv = pd.merge(csv2, csv1_selected, on='user_id', how='left')

# 結果を確認
# print(merged_csv)

# 必要に応じてCSVに保存
merged_csv.to_csv('BIGBERT.csv', index=False)
