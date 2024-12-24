import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

# 残差の標準偏差を含むCSVファイルを読み込む
residuals_df = pd.read_csv('INTERACTION_residuals_std_results.csv')

# 元のデータセットのCSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# group_c列を作成（group列がgroupcの場合は1、それ以外は0）
df['group_c'] = (df['group'] == 'groupc').astype(int)
# group_flag列を追加 (group列がgroupcの場合は1、それ以外は0)
df['group_flag'] = (df['group'] == 'groupc').astype(int)

# 説明変数と従属変数の指定
# explanatory_variable = 'ave_cos_BERT_diary_Human_count'  # 説明変数
explanatory_variable = 'ave_cos_BERT_diary_Human_count' 
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

# 説明変数とgroup_flagを標準化
scaler = StandardScaler()
# df[['ave_cos_BERT_diary_Human_count', 'group_flag']] = scaler.fit_transform(df[['ave_cos_BERT_diary_Human_count', 'group_flag']])
df[['ave_cos_BERT_diary_Human_count', 'group_c']] = scaler.fit_transform(df[['ave_cos_BERT_diary_Human_count', 'group_c']])

# 交互作用項を作成
df['interaction'] = df[explanatory_variable] * df['group_c']

def run_robust_regression(dependent_var):
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

    # 説明変数（ave_cos_BERT_diary_Human_count, group_c, interaction）を含むデザイン行列を作成
    X = sm.add_constant(df[['ave_cos_BERT_diary_Human_count', 'group_c', 'interaction']])

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
    plt.scatter(df['ave_cos_BERT_diary_Human_count'], y, label='Data')
    plt.plot(df['ave_cos_BERT_diary_Human_count'], model.predict(X), color='red', label='Fit')

    # 軸のフォントサイズの設定
    font_size = 20  # 任意のフォントサイズ
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # タイトルとラベルの設定
    plt.title(f'Robust Regression: {dependent_var} ~ {explanatory_variable} + group_c + interaction')
    plt.xlabel('Standardized ave_cos_BERT_diary_Human_count')
    plt.ylabel(dependent_var)

    plt.xlim(-0.02, 1.02)  # X軸の範囲
    plt.yticks(np.arange(1.0, 7.0, 1.0))
    plt.ylim(0.9, 6.1)  # Y軸の範囲
    plt.legend()
    plt.grid(False)

    # SVGファイルとして保存
    plt.savefig(f"interaction_SVGs/{dependent_var}_regression_plot.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 各従属変数に対してロバスト回帰を実行
for dependent_var in dependent_variables:
    run_robust_regression(dependent_var)
