import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

# 残差の標準偏差を含むCSVファイルを読み込む
residuals_df = pd.read_csv('all_SD_residuals_std_results.csv')

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

# 統制変数を指定
df['group_c'] = pd.get_dummies(df['group'], drop_first=True).astype(int)
control_variables = ['group_c']  # 統制変数リスト

# 3SDで外れ値を除外する関数
def remove_outliers(df, columns):
    """指定された列で3SDによる外れ値除外を行う"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# すべての変数（説明変数と従属変数）を含む列で3SDカットを実施
all_variables = [explanatory_variable] + dependent_variables
df = remove_outliers(df, all_variables)

# ロバスト回帰の関数
def run_robust_regression_with_outlier_removal(dependent_var, control_vars=[]):
    """3SDで外れ値を除外した後にロバスト回帰を実行する関数"""
    
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
    X = sm.add_constant(df[explanatory_variable])

    # ロバスト回帰の実行（Huber’s T normを使用）
    model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

    # ロバスト回帰の結果表示
    print(f'Robust Regression results for {dependent_var}:')
    print(model.summary())

    # 残差の計算
    residuals = model.resid

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

    # 軸と範囲の設定
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y軸の範囲
    plt.grid(False)

    # SVGファイルとして保存
    plt.savefig(f"SVGs/{dependent_var}_SD_regression_plot_after_3SD.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 各従属変数に対してロバスト回帰を実行（3SDで外れ値を除外した後）
for dependent_var in dependent_variables:
    run_robust_regression_with_outlier_removal(dependent_var, control_vars=control_variables)
