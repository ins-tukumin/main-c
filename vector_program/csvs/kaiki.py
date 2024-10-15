import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('BIGDATA.csv')

# 説明変数と従属変数の指定
explanatory_variable = df['ave_cos_diary_Human']  # 説明変数（例: 'age'など）
dependent_variables = ['ave_understanding','ave_PANAS_P','ave_PANAS_N','ave_competence','ave_warmth','ave_satisfaction','ave_effectiveness','ave_efficiency','ave_willingness']  # 複数の従属変数

# 説明変数に定数項を追加（回帰分析のため）
X = sm.add_constant(explanatory_variable)

# 各従属変数に対して回帰分析とプロットを行う
for dependent_var in dependent_variables:
    # 従属変数の選択
    y = df[dependent_var]
    
    # 回帰分析の実行
    model = sm.OLS(y, X).fit()
    print(f'Regression results for {dependent_var}:')
    print(model.summary())  # 回帰結果の表示
    
    # 回帰直線を引くための予測値
    predictions = model.predict(X)
    
    # プロットの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(explanatory_variable, y, label='Data Points')
    plt.plot(explanatory_variable, predictions, color='red', label='Regression Line')
    plt.title(f'Regression: {dependent_var} ~ Human-Diary')
    plt.xlabel('Human-Diary')
    plt.ylabel(dependent_var)
    plt.legend()
    plt.grid(True)
    plt.show()
