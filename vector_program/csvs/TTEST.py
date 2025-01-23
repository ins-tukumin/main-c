import pandas as pd
from scipy.stats import ttest_ind, levene

# CSVを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupaとgroupcのデータを抽出
groupa = df[df['group'] == 'groupa']
groupc = df[df['group'] == 'groupc']

# 比較する列のリスト
columns_to_test = [#'ave_PANAS_P', 'ave_PANAS_N',
    #'ave_competence', 'ave_warmth',
    #'ave_willingness', 'ave_understanding'
    'ave_cos_diary_Human', 'stan_topic_count']  # 指定した列をここに記載

# 結果を格納するリスト
results = []

# 各列についてループ
for col in columns_to_test:
    # groupaとgroupcの値を取得
    values_a = groupa[col].dropna()
    values_c = groupc[col].dropna()
    
    # 等分散性をLevene検定で確認
    _, p_levene = levene(values_a, values_c)
    equal_var = p_levene > 0.05  # p値 > 0.05 なら等分散を仮定

    # t検定を実行
    t_stat, p_value = ttest_ind(values_a, values_c, equal_var=equal_var)
    
    # 結果を保存
    results.append({
        'column': col,
        't_stat': t_stat,
        'p_value': p_value,
        'equal_var_assumed': equal_var
    })

# 結果をDataFrameに変換して出力
results_df = pd.DataFrame(results)
print(results_df)

# 必要に応じてCSVとして保存
results_df.to_csv('t_test_results.csv', index=False)
