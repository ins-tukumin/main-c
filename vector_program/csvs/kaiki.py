import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('BIGDATA.csv')

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_diary_Human'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N'
    #'ave_competence',
    #'ave_warmth', 'ave_willingness','ave_understanding'
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
        plt.scatter(group_data[explanatory_variable], y, label='Data Points')
        plt.plot(group_data[explanatory_variable], predictions, color='red', label='Regression Line')

        # タイトルとラベルの設定
        plt.title(f'Regression: {dependent_var} ~ {explanatory_variable} (Group: {group})')
        # plt.xlabel('Human-Diary')
        # plt.ylabel(dependent_var)

        # 軸のスケールを指定 (例: 0から1までの範囲)
        plt.xlim(0, 0.4)  # X軸の範囲
        plt.ylim(1, 6)  # Y軸の範囲

        plt.legend()
        plt.grid(True)
        plt.show()
