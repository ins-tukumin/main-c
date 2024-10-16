import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# データの読み込み
df = pd.read_csv('BIGDATA.csv')

# 説明変数と従属変数を指定
explanatory_variable = 'ave_cos_diary_Human'  # 説明変数
dependent_variables = [
    'ave_understanding', 'ave_PANAS_P', 'ave_PANAS_N', 'ave_competence',
    'ave_warmth', 'ave_satisfaction', 'ave_effectiveness',
    'ave_efficiency', 'ave_willingness'
]  # 従属変数のリスト

# 各従属変数に対して線形混合モデルを実行
for dependent_var in dependent_variables:
    # 説明変数と定数項を追加
    X = sm.add_constant(df[explanatory_variable])

    # 線形混合モデルの構築（変量効果: group）
    model = MixedLM(df[dependent_var], X, groups=df['group'])
    result = model.fit()

    # モデルの結果を表示
    print(f'Results for {dependent_var}:')
    print(result.summary())
