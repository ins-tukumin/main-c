import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.stattools import omni_normtest
from scipy.stats import shapiro

# 残差の標準偏差を含むCSVファイルを読み込む
residuals_df = pd.read_csv('all_residuals_std_results.csv')

# 元のデータセットのCSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

df['group_c'] = pd.get_dummies(df['group'], drop_first=True).astype(int)
# 統制変数を指定
control_variables = ['group_c']  # 統制変数リスト
print(df)

def run_robust_regression(dependent_var, control_vars=[]):
    """指定された従属変数に対してロバスト回帰を実行し、結果を出力する関数"""
    
    # 従属変数の選択
    y = df[dependent_var]

    # 残差の標準偏差を取得
    residuals_std = residuals_df.loc[
        residuals_df['dependent_var'] == dependent_var, 
        'residuals_std'
    ].values
    
    if len(residuals_std) == 0:
        print(f"No residuals_std found for dependent variable {dependent_var}.")
        return
    
    residuals_std = residuals_std[0]  # 1つの値を取得
    
    # Huberの閾値を設定
    delta = 1.345 * residuals_std
    print(f'Using Huber threshold (delta) for dependent variable {dependent_var}: {delta}')

    # 説明変数に定数項を追加（回帰分析のため）
    # X = sm.add_constant(df[[explanatory_variable] + control_vars])
    X = sm.add_constant(df[explanatory_variable])

    # ロバスト回帰の実行（Huber’s T normを使用）
    model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

    # ロバスト回帰の結果表示
    print(f'Robust Regression results for {dependent_var}:')
    print(model.summary())

    # 残差の計算
    residuals = model.resid

    # 残差の検定
    #jb_test = jarque_bera(residuals)
    #print(f'Jarque-Bera test for residuals of {dependent_var}:')
    #print(f'Statistic: {jb_test.statistic}, p-value: {jb_test.pvalue}')

    #omni_test = omni_normtest(residuals)
    #print(f'Omnibus test for residuals of {dependent_var}:')
    #print(f'Statistic: {omni_test.statistic}, p-value: {omni_test.pvalue}')

    # Shapiro-Wilk検定の実行
    shapiro_test_stat, shapiro_p_value = shapiro(residuals)

    # 検定結果の出力
    print(f'Shapiro-Wilk Test for {dependent_var}:')
    print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(df[explanatory_variable], y)
    plt.plot(df[explanatory_variable], model.predict(X), color='red')

    # 軸のフォントサイズの設定
    font_size = 20  # 任意のフォントサイズ
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # タイトルとラベルの設定
    # plt.title(f'Robust Regression: {dependent_var} ~ {explanatory_variable}')
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 7.0, 1.0))
    plt.ylim(0.9, 6.1)  # Y軸の範囲
    plt.legend()
    plt.grid(False)

    # SVGファイルとして保存
    plt.savefig(f"SVGs/{dependent_var}_regression_plot.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 各従属変数に対してロバスト回帰を実行
for dependent_var in dependent_variables:
    run_robust_regression(dependent_var, control_vars=control_variables)
