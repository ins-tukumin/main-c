import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

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
    'ave_competence',
    'ave_warmth', 'ave_willingness','ave_understanding'
]  # 複数の従属変数

# 全データを使って、各従属変数に対して混合効果モデルとプロットを行う
for dependent_var in dependent_variables:
    # プロットの作成
    plt.figure(figsize=(8, 6))

    # グループごとに異なる色で回帰分析と回帰直線の描画
    unique_groups = df['group'].unique()
    colors = ['blue', 'green', 'orange']  # グループごとの色

    for group, color in zip(unique_groups, colors):
        # グループごとにデータをフィルタリング
        group_data = df[df['group'] == group]

        # モデルの式を定義
        formula = f"{dependent_var} ~ {explanatory_variable}"
        
        # グループごとに混合効果モデル（ここでは固定効果として扱う）
        model = smf.ols(formula, data=group_data).fit()

        # 回帰結果の表示
        print(f'Regression results for {dependent_var} (Group: {group}):')
        print(model.summary())

        # 回帰直線を引くための予測値
        predictions = model.predict(group_data[explanatory_variable])

        # 残差の計算
        residuals = model.resid
        residuals_std = np.std(residuals)  # 残差の標準偏差の計算

        # 残差の標準偏差をリストに追加
        results_list.append({
            'dependent_var': dependent_var,
            'group': group,
            'residuals_std': residuals_std
        })

        # グループごとの回帰直線をプロット
        plt.plot(group_data[explanatory_variable], predictions, color=color, label=f'Group {group} Regression Line')
        
        # グループごとのデータポイントもプロット（オプション）
        plt.scatter(group_data[explanatory_variable], group_data[dependent_var], color=color, alpha=0.5)

    # タイトルとラベルの設定
    plt.title(f'Regression: {dependent_var} ~ {explanatory_variable}')
    plt.xlabel('Human-Diary')
    plt.ylabel(dependent_var)
    plt.xlim(0.3, 0.8)  # X軸の範囲
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.ylim(0.9, 5.1)  # Y軸の範囲
    plt.legend()  # 凡例を表示
    plt.grid(False)
    plt.savefig(f"SVGs/{dependent_var}_grouped_regression_plot.svg", format="svg")
    plt.close()  # プロットを閉じてメモリを解放

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_list)

# 残差の標準偏差をCSVファイルに保存
# results_df.to_csv('all_residuals_std_results.csv', index=False)
