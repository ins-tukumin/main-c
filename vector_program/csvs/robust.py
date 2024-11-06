import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.robust.norms as norms  # ノルム関数をここからインポート
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.stattools import omni_normtest

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = ['ave_PANAS_P', 'ave_PANAS_N', 'ave_competence','ave_warmth', 'ave_willingness','ave_understanding']  # 複数の従属変数

# グループごとのデータを取得
groups = df['group'].unique()

# 各groupごと、各従属変数に対してロバスト回帰と分位点回帰を実行
for group in groups:
    # グループごとのデータをフィルタリング
    group_data = df[df['group'] == group]

    # 説明変数に定数項を追加（回帰分析のため）
    X = sm.add_constant(group_data[explanatory_variable])

    for dependent_var in dependent_variables:
        # 従属変数の選択
        y = group_data[dependent_var]

        # ロバスト回帰の実行（Huber’s T normを使用）
        model = sm.RLM(y, X, M=norms.HuberT()).fit()

        # ロバスト回帰の結果表示
        print(f'Robust Regression results for {group} - {dependent_var}:')
        print(model.summary())

        # ロバスト回帰の予測値
        predictions = model.predict(X)

        # 残差の計算
        residuals = y - predictions

        # Jarque-Bera検定
        jb_test = jarque_bera(residuals)
        print(f'Jarque-Bera test for residuals of {dependent_var} (Group: {group}):')
        print(f'Statistic: {jb_test.statistic}, p-value: {jb_test.pvalue}')

        # Omnibus検定
        omni_test = omni_normtest(residuals)
        print(f'Omnibus test for residuals of {dependent_var} (Group: {group}):')
        print(f'Statistic: {omni_test.statistic}, p-value: {omni_test.pvalue}')

        # 残差の標準偏差
        residuals_std = np.std(residuals)
        print(f'Standard deviation of residuals for {dependent_var} (Group: {group}): {residuals_std}\n')

        # AIC/BICの計算
        n = len(y)  # 観測数
        rss = np.sum(residuals**2)  # 残差平方和
        k = model.params.shape[0]  # モデルのパラメータ数

        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        
        print(f'AIC for {dependent_var} (Group: {group}): {aic}')
        print(f'BIC for {dependent_var} (Group: {group}): {bic}\n')
        # Huber損失の計算
        delta = 1.345  # Huber損失で使用する閾値
        huber_loss = np.where(np.abs(residuals) <= delta, 
                              0.5 * residuals**2, 
                              delta * (np.abs(residuals) - 0.5 * delta)).mean()
        print(f'Huber loss for {dependent_var} (Group: {group}): {huber_loss}\n')

        # プロットの作成
        plt.grid(False)
        plt.figure(figsize=(8, 6))
        # plt.scatter(group_data[explanatory_variable], y, label='Data Points')
        # plt.plot(group_data[explanatory_variable], predictions, color='red', label='Robust Regression Line')
        plt.scatter(group_data[explanatory_variable], y)
        plt.plot(group_data[explanatory_variable], predictions, color='red')

        # タイトルとラベルの設定
        plt.title(f'Robust Regression: {dependent_var} ~ {explanatory_variable} (Group: {group})')

        # 軸のスケールを指定
        plt.xlim(0.3, 0.8)  # X軸の範囲
        plt.yticks(np.arange(1.0, 7.0, 1.0))
        plt.ylim(0.9, 6.1)  # Y軸の範囲

        plt.legend()
        plt.grid(False)
        plt.show()

        # SVGファイルとして保存
        plt.savefig(f"SVGs/{dependent_var}_regression_plot.svg", format="svg")
