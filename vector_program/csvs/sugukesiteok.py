import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def remove_ave_prefix(string):
    if string.startswith("ave_"):
        return string[4:]  # "ave_"の長さは4なので、それ以降を取得
    return string  # "ave_"がない場合はそのまま返す

# CSVファイルを読み込む
df = pd.read_csv('BIGDATA.csv')

# 説明変数と従属変数の指定
explanatory_variable_name = 'ave_cos_diary_Human'
dependent_variables = ['ave_PANAS_P', 'ave_PANAS_N', 'ave_competence', 
                       'ave_warmth', 'ave_willingness', 'ave_understanding']  # 複数の従属変数

# グループごとにデータを分割
groups = df['group'].unique()  # group列の種類を取得

for group in groups:
    # グループごとのデータを抽出
    group_df = df[df['group'] == group]

    # 説明変数に定数項を追加（回帰分析のため）
    X = sm.add_constant(group_df[explanatory_variable_name])

    # グラフのレイアウト設定（2行3列のサブプロット）
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()  # 2次元から1次元に展開

    print(f'\nGroup: {group}\n' + '-' * 40)

    # 各従属変数に対して回帰分析とプロットを行う
    for i, dependent_var in enumerate(dependent_variables):
        # 従属変数の選択
        y = group_df[dependent_var]

        # 回帰分析の実行
        model = sm.OLS(y, X).fit()
        print(f'Regression results for {dependent_var} in group {group}:')
        print(model.summary())  # 回帰結果の表示

        # 回帰直線を引くための予測値
        predictions = model.predict(X)

        # プロットの作成
        ax = axes[i]
        ax.scatter(group_df[explanatory_variable_name], y, label='Data Points')
        ax.plot(group_df[explanatory_variable_name], predictions, color='red', label='Regression Line')

        result = remove_ave_prefix(dependent_var)
        ax.set_title(f'{result}', fontsize=18)

        # Y軸の範囲と目盛を条件分岐
        if dependent_var in ['ave_PANAS_P', 'ave_PANAS_N']:
            ax.set_ylim(1, 6)
            ax.set_yticks(np.arange(1, 7, 1.0))
        else:
            ax.set_ylim(1, 5)
            ax.set_yticks(np.arange(1, 6, 1.0))

        # X軸の範囲と目盛を設定
        ax.set_xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))

        # 軸のフォントサイズを設定
        ax.tick_params(axis='both', labelsize=18)

        # 凡例のフォントサイズを設定
        ax.legend(fontsize=20)

        # グリッド表示
        ax.grid(True)

    # レイアウト調整と表示
    plt.tight_layout()
    plt.show()
