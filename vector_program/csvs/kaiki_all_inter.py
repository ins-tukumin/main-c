import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# group_flag列を追加 (group列がgroupcの場合は1、それ以外は0)
df['group_flag'] = (df['group'] == 'groupc').astype(int)

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human' 
# explanatory_variable = 'stan_topic_count'  # 元の説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence',
    'ave_warmth', 'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

# 説明変数とgroup_flagを標準化
scaler = StandardScaler()
# df[['stan_topic_count', 'group_flag']] = scaler.fit_transform(df[['stan_topic_count', 'group_flag']])
df[['ave_cos_BERT_diary_Human', 'group_flag']] = scaler.fit_transform(df[['ave_cos_BERT_diary_Human', 'group_flag']])

# 交互作用項を追加
df['interaction'] = df['ave_cos_BERT_diary_Human'] * df['group_flag']

# 結果を格納するリスト
results_list = []

# 重回帰分析を行う
for dependent_var in dependent_variables:
    # 従属変数の選択
    y = df[dependent_var]

    # 説明変数に定数項を追加（重回帰分析のため）
    X = sm.add_constant(df[['ave_cos_BERT_diary_Human', 'group_flag', 'interaction']])

    # 回帰分析の実行
    model = sm.OLS(y, X).fit()

    # 回帰結果の表示
    print(f'Regression results for {dependent_var}:')
    print(model.summary())

    # 回帰直線を引くための予測値
    predictions = model.predict(X)

    # 残差の計算
    residuals = model.resid
    residuals_std = np.std(residuals)  # 残差の標準偏差の計算

    # Shapiro-Wilk検定の実行
    shapiro_test_stat, shapiro_p_value = shapiro(residuals)

    # 検定結果の出力
    print(f'Shapiro-Wilk Test for {dependent_var}:')
    print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

    # 残差の標準偏差をリストに追加
    results_list.append({
        'dependent_var': dependent_var,
        'residuals_std': residuals_std
    })

    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(df['ave_cos_BERT_diary_Human'], y, label='Data')
    plt.plot(df['ave_cos_BERT_diary_Human'], predictions, color='red', label='Fit')

    # タイトルとラベルの設定
    plt.title(f'Regression: {dependent_var} ~ Standardized Variables')
    plt.xlabel('Standardized Human-Diary')
    plt.ylabel(dependent_var)

    plt.xlim(-3, 3)  # X軸の範囲
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y軸の範囲
    plt.legend()
    plt.grid(False)
    # plt.show()
    # plt.savefig(f"topic_SVGs/{dependent_var}_regression_plot.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_list)

# 残差の標準偏差をCSVファイルに保存
results_df.to_csv('INTERACTION_residuals_std_results.csv', index=False)
