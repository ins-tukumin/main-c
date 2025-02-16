import pandas as pd
from scipy.stats import ttest_ind, levene

# CSVファイルを読み込む
df = pd.read_csv("BERT.csv")

# group列で 'groupa' と 'groupc' のデータのみを抽出
group_a = df[df['group'] == 'c']
group_c = df[df['group'] == 'rep']

# t検定を行う列を指定
target_columns = ["ave_BERT_cos_AI_Human", "ave_BERT_cos_diary_AI", "ave_cos_BERT_diary_Human"]

# t検定の結果を保存するための辞書
t_test_results = {}

# 各指定列で等分散性を検定し、適切なt検定を実行
for column in target_columns:
    # 'groupa' と 'groupc' のデータを取得
    data_a = group_a[column].dropna()  # 欠損値を除去
    data_c = group_c[column].dropna()  # 欠損値を除去

    # 等分散性の検定（Levene検定）
    levene_stat, levene_p = levene(data_a, data_c)

    # Levene検定の結果に基づいてt検定を選択
    if levene_p < 0.05:
        # 分散が等しくない場合はWelchのt検定
        t_stat, p_value = ttest_ind(data_a, data_c, equal_var=False)
        test_type = "Welch's t-test"
        
        # Welchの自由度を計算
        s1_squared = data_a.var(ddof=1)  # グループaの分散
        s2_squared = data_c.var(ddof=1)  # グループcの分散
        n1 = len(data_a)  # グループaのサンプルサイズ
        n2 = len(data_c)  # グループcのサンプルサイズ
        df = ((s1_squared / n1 + s2_squared / n2) ** 2) / \
             ((s1_squared / n1) ** 2 / (n1 - 1) + (s2_squared / n2) ** 2 / (n2 - 1))
    else:
        # 分散が等しい場合は学生のt検定
        t_stat, p_value = ttest_ind(data_a, data_c, equal_var=True)
        test_type = "Student's t-test"
        
        # Student's自由度を計算
        n1 = len(data_a)
        n2 = len(data_c)
        df = n1 + n2 - 2

    # 結果を辞書に保存
    t_test_results[column] = {
        'test_type': test_type,
        'levene_p_value': levene_p,
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': df
    }

# 結果を表示
for column, result in t_test_results.items():
    print(f"Column: {column}")
    print(f"Test Type: {result['test_type']}")
    print(f"Levene's test p-value: {result['levene_p_value']}")
    print(f"t-statistic: {result['t_statistic']}, p-value: {result['p_value']}")
    print(f"Degrees of Freedom: {result['degrees_of_freedom']}\n")
