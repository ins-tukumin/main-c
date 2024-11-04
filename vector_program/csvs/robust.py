import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.robust.norms as norms  # ノルム関数をここからインポート
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = ['ave_PANAS_P', 'ave_PANAS_N', 'ave_competence','ave_warmth', 'ave_satisfaction', 'ave_effectiveness','ave_efficiency', 'ave_willingness','ave_understanding']  # 複数の従属変数

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

        # プロットの作成
        plt.figure(figsize=(8, 6))
        plt.scatter(group_data[explanatory_variable], y, label='Data Points')
        plt.plot(group_data[explanatory_variable], predictions, color='red', label='Robust Regression Line')

        # タイトルとラベルの設定
        plt.title(f'Robust Regression: {dependent_var} ~ {explanatory_variable} (Group: {group})')

        # 軸のスケールを指定
        plt.xlim(0.4, 1.0)  # X軸の範囲
        plt.ylim(1, 5)  # Y軸の範囲

        plt.legend()
        plt.grid(True)
        plt.show()
