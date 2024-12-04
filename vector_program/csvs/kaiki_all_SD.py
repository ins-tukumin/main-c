import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# 説明変数と従属変数の指定
explanatory_variable = 'stan_topic_count'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence',
    'ave_warmth', 'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

# 説明変数と従属変数のデータをコピー（元のデータを保護）
data = df[[explanatory_variable] + dependent_variables].copy()

# 3SDで外れ値を除外する関数を定義
def remove_outliers_3sd(series):
    mean = series.mean()
    std = series.std()
    return series[(series >= mean - 3 * std) & (series <= mean + 3 * std)]

# 説明変数に対して3SDで外れ値を除外
data[explanatory_variable] = remove_outliers_3sd(data[explanatory_variable])

# 各従属変数に対して3SDで外れ値を除外
for var in dependent_variables:
    data[var] = remove_outliers_3sd(data[var])

# 欠損値（外れ値として除外されたデータ）を含む行を削除
data.dropna(inplace=True)

# 結果を格納するリスト
results_list = []

# 説明変数に定数項を追加（回帰分析のため）
X = sm.add_constant(data[explanatory_variable])

# 全データを使って、各従属変数に対して回帰分析とプロットを行う
for dependent_var in dependent_variables:
    # 従属変数の選択
    y = data[dependent_var]

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
    plt.scatter(data[explanatory_variable], y)
    plt.plot(data[explanatory_variable], predictions, color='red')

    # タイトルとラベルの設定
    plt.title(f'Regression: {dependent_var} ~ {explanatory_variable}')
    plt.xlabel('Human-Diary')
    plt.ylabel(dependent_var)

    # 軸の範囲と目盛りの設定
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y軸の範囲

    # 凡例とグリッドの設定
    plt.legend()
    plt.grid(False)

    # プロットの保存（必要に応じてコメントアウトを外してください）
    # plt.savefig(f"SVGs/{dependent_var}_regression_plot.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_list)

# 残差の標準偏差をCSVファイルに保存
results_df.to_csv('all_SD_residuals_std_results.csv', index=False)
