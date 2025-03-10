import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# 結果を格納するリスト
results_list = []

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

# 外れ値を削除する関数 (2SDまたは3SD)
def remove_outliers_sd(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 各変数で外れ値を削除
variables = [explanatory_variable] + dependent_variables
for var in variables:
    df = remove_outliers_sd(df, var, threshold=3)  # 3SDを基準に外れ値を削除

# 説明変数に定数項を追加（回帰分析のため）
X = sm.add_constant(df[explanatory_variable])

# 各従属変数に対して回帰分析とプロットを行う
for dependent_var in dependent_variables:
    # 従属変数の選択
    y = df[dependent_var]

    # 回帰分析の実行
    model = sm.OLS(y, X).fit()

    # 回帰結果の表示
    print(f'Regression results for {dependent_var}:')
    print(model.summary())

    # 回帰直線を引くための予測値
    predictions = model.predict(X)

    # 残差の計算
    residuals = model.resid
    residuals_std = np.std(residuals)

    # Shapiro-Wilk検定の実行
    shapiro_test_stat, shapiro_p_value = shapiro(residuals)

    # 検定結果の出力
    print(f'Shapiro-Wilk Test for {dependent_var}:')
    print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

    # 結果リストに追加
    results_list.append({
        'dependent_var': dependent_var,
        'residuals_std': residuals_std
    })

    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(df[explanatory_variable], y)
    plt.plot(df[explanatory_variable], predictions, color='red')

    # タイトルとラベルの設定
    plt.title(f'Regression: {dependent_var} ~ {explanatory_variable}')
    plt.xlabel('Human-Diary')
    plt.ylabel(dependent_var)

    # 軸の範囲設定
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y軸の範囲
    plt.legend()
    plt.grid(False)

    # プロット保存
    #plt.savefig(f"SVGs/{dependent_var}_regression_plot.svg", format="svg")
    plt.close()

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_list)

# 残差の標準偏差をCSVファイルに保存
# results_df.to_csv('all_residuals_std_results.csv', index=False)
