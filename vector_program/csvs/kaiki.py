import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence',
    'ave_warmth', 'ave_willingness','ave_understanding',
    'ave_satisfaction', 'ave_effectiveness','ave_efficiency'
]  # 複数の従属変数
# 'ave_PANAS_P', 'ave_PANAS_N', 'ave_competence','ave_warmth', 'ave_satisfaction', 'ave_effectiveness','ave_efficiency', 'ave_willingness','ave_understanding'

# groupごとのデータを取得
groups = df['group'].unique()  # group列のユニークな値を取得

# 各groupごと、各従属変数に対して回帰分析とプロットを行う
for group in groups:
    # groupごとにデータをフィルタリング
    group_data = df[df['group'] == group]
    
    # 説明変数に定数項を追加（回帰分析のため）
    X = sm.add_constant(group_data[explanatory_variable])

    for dependent_var in dependent_variables:
        # 従属変数の選択
        y = group_data[dependent_var]

        # 回帰分析の実行
        model = sm.OLS(y, X).fit()

        # 回帰結果の表示
        print(f'Regression results for {group} - {dependent_var}:')
        print(model.summary())

        # 回帰直線を引くための予測値
        predictions = model.predict(X)

        # プロットの作成
        plt.figure(figsize=(8, 6))
        plt.scatter(group_data[explanatory_variable], y)
        plt.plot(group_data[explanatory_variable], predictions, color='red')

        # タイトルとラベルの設定
        plt.title(f'Regression: {dependent_var} ~ {explanatory_variable} (Group: {group})')
        # plt.xlabel('Human-Diary')
        # plt.ylabel(dependent_var)

        # 軸のスケールを指定 (例: 0から1までの範囲)
        # plt.xlim(0.4, 1.0)  # X軸の範囲
        # plt.ylim(1, 5)  # Y軸の範囲

        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        # 軸のスケールを指定
        plt.xlim(0.3, 0.8)  # X軸の範囲
        plt.yticks(np.arange(1.0, 6.0, 1.0))
        plt.ylim(0.9, 5.1)  # Y軸の範囲

        plt.legend()
        plt.grid(False)
        # plt.show()

        # SVGファイルとして保存
        plt.savefig(f"SVGs/{dependent_var}_regression_plot.svg", format="svg")
