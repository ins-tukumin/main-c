import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('hyoutei.csv')

# user_idごとにグループ化して、day列に基づいて各日付のデータをまとめる
result = []

# 'user_id'列を基にグループ化
for user_id, group in df.groupby('user_id'):
    # 空の辞書を作成して、まとめるデータを保存
    user_data = {'user_id': user_id}
    
    # dayごとのデータをまとめる
    for day in range(1, 4):  # day1, day2, day3 を処理
        day_group = group[group['day'] == day]
        for column in ['understanding', 'PANAS_P', 'PANAS_N', 'competence', 'warmth', 'satisfaction', 'effectiveness', 'efficiency', 'willingness']:
            user_data[f'day{day}_{column}'] = day_group[column].values[0] if not day_group.empty else None
    
    # day1, day2, day3の平均を計算
    for column in ['understanding', 'PANAS_P', 'PANAS_N', 'competence', 'warmth', 'satisfaction', 'effectiveness', 'efficiency', 'willingness']:
        user_data[f'ave_{column}'] = group[column].mean().round(3)
    
    # まとめたuser_dataをリストに追加
    result.append(user_data)

# リストをDataFrameに変換
result_df = pd.DataFrame(result)

# 結果を確認
print(result_df.head())

# 新しいCSVファイルとして保存
result_df.to_csv('transformed_data.csv', index=False)
