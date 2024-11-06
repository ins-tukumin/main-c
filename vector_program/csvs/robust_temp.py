import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms  # ノルム関数をここからインポート
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.stattools import omni_normtest

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = ['ave_PANAS_P', 'ave_PANAS_N', 'ave_competence', 'ave_warmth', 'ave_willingness', 'ave_understanding']  # 複数の従属変数

# グループごとのデータを取得
groups = df['group'].unique()

# 結果を格納するためのリスト
results_list = []

# 各groupごと、各従属変数に対してロバスト回帰を実行
for group in groups:
    # グループごとのデータをフィルタリング
    group_data = df[df['group'] == group]

    # 説明変数に定数項を追加（回帰分析のため）
    X = sm.add_constant(group_data[explanatory_variable])

    for dependent_var in dependent_variables:
        # 従属変数の選択
        y = group_data[dependent_var]

        # tの値を0.2から2.0まで0.1刻みでループ
        for t in np.arange(0.01, 10, 0.01):
            # ロバスト回帰の実行（Huber’s T normを使用）
            model = sm.RLM(y, X, M=norms.HuberT(t=t)).fit()

            # AIC/BICの計算
            n = len(y)  # 観測数
            rss = np.sum((y - model.predict(X)) ** 2)  # 残差平方和
            k = model.params.shape[0]  # モデルのパラメータ数

            aic = round(n * np.log(rss / n) + 2 * k, 3)
            bic = round(n * np.log(rss / n) + k * np.log(n), 3)
            
            # Huber損失の計算
            residuals = y - model.predict(X)
            huber_loss = round(np.where(np.abs(residuals) <= t, 
                                         0.5 * residuals**2, 
                                         t * (np.abs(residuals) - 0.5 * t)).mean(), 3)

            # 結果をリストに追加（tの値をキーとして、横に並べる）
            results_list.append({
                'group': group,
                'dependent_var': dependent_var,
                't_value': round(t, 1),
                'AIC': aic,
                'BIC': bic,
                'Huber_loss': huber_loss
            })

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_list)

# 各グループごとにプロットを作成
for group in groups:
    plt.figure(figsize=(15, 10))
    
    # グループのデータをフィルタリング
    group_results = results_df[results_df['group'] == group]
    
    # AICのプロット
    plt.subplot(3, 1, 1)
    for dependent_var in dependent_variables:
        subset = group_results[group_results['dependent_var'] == dependent_var]
        plt.plot(subset['t_value'], subset['AIC'], marker='o', label=dependent_var)
    plt.title(f'AIC vs t for Group: {group}')
    plt.xlabel('t value')
    plt.ylabel('AIC')
    plt.legend()
    plt.grid()

    # BICのプロット
    plt.subplot(3, 1, 2)
    for dependent_var in dependent_variables:
        subset = group_results[group_results['dependent_var'] == dependent_var]
        plt.plot(subset['t_value'], subset['BIC'], marker='o', label=dependent_var)
    plt.title(f'BIC vs t for Group: {group}')
    plt.xlabel('t value')
    plt.ylabel('BIC')
    plt.legend()
    plt.grid()

    # Huber損失のプロット
    plt.subplot(3, 1, 3)
    for dependent_var in dependent_variables:
        subset = group_results[group_results['dependent_var'] == dependent_var]
        plt.plot(subset['t_value'], subset['Huber_loss'], marker='o', label=dependent_var)
    plt.title(f'Huber Loss vs t for Group: {group}')
    plt.xlabel('t value')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.grid()

    # プロットを表示
    plt.tight_layout()
    plt.savefig(f'Group_{group}_results_plot.svg', format='svg')  # プロットをSVGとして保存
    plt.show()
