import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, norm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 残差の標準偏差を含むCSVファイルを読み込む
residuals_df = pd.read_csv('INTERACTION_residuals_std_results.csv')

# 元のデータセットのCSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# groupbを削除
df = df[df['group'] != 'groupb']

# group_c列を作成（group列がgroupcの場合は1、それ以外は0）
df['group_c'] = (df['group'] == 'groupc').astype(int)

# 説明変数と従属変数の指定
explanatory_variable = 'ave_cos_BERT_diary_Human'  # 説明変数
dependent_variables = [
    'ave_PANAS_P', 'ave_PANAS_N',
    'ave_competence', 'ave_warmth',
    'ave_willingness', 'ave_understanding'
]  # 複数の従属変数

# 説明変数とgroup_cを標準化
scaler = StandardScaler()
df[['ave_cos_BERT_diary_Human', 'group_c']] = scaler.fit_transform(df[['ave_cos_BERT_diary_Human', 'group_c']])

# 交互作用項を作成
df['interaction'] = df[explanatory_variable] * df['group_c']

def run_robust_regression_with_vif_f_r2_jn(dependent_var):
    """指定された従属変数に対してロバスト回帰、VIF、F値、擬似R²を出力し、交互作用項が有意な場合にジョンソン-ネイマン法を実行する関数"""
    
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

    # 説明変数（ave_cos_BERT_diary_Human, group_c, interaction）を含むデザイン行列を作成
    X = sm.add_constant(df[['ave_cos_BERT_diary_Human', 'group_c', 'interaction']])

    # VIFの計算
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f'VIF for {dependent_var}:')
    print(vif_data, "\n")

    # ロバスト回帰の実行（Huber’s T normを使用）
    robust_model = sm.RLM(y, X, M=norms.HuberT(t=delta)).fit()

    # ロバスト回帰の結果表示
    print(f'Robust Regression results for {dependent_var}:')
    print(robust_model.summary())

    # 通常のOLSモデルを使用してF値を計算
    ols_model = sm.OLS(y, X).fit()
    print(f"OLS F-value for {dependent_var}: {ols_model.fvalue:.2f}, p-value: {ols_model.f_pvalue:.4f}")

    # R²の計算（擬似R²を計算）
    total_variance = np.var(y, ddof=1)
    residual_variance = robust_model.scale
    pseudo_r2 = 1 - (residual_variance / total_variance)
    print(f"Pseudo R² for {dependent_var}: {pseudo_r2:.3f}\n")

    # 残差の計算
    residuals = robust_model.resid

    # Shapiro-Wilk検定の実行
    shapiro_test_stat, shapiro_p_value = shapiro(residuals)

    # 検定結果の出力
    print(f'Shapiro-Wilk Test for {dependent_var}:')
    print(f'Statistic: {shapiro_test_stat}, p-value: {shapiro_p_value}\n')

    # ジョンソン-ネイマン法の実行（交互作用項が有意な場合）
    interaction_p_value = robust_model.pvalues['interaction']
    if interaction_p_value < 0.05:
        print("\n交互作用が有意です。ジョンソン-ネイマン法を実行します。\n")
        beta = robust_model.params
        se = robust_model.bse

        # Simple Slopeの計算
        z_values = np.linspace(df['group_c'].min(), df['group_c'].max(), 100)
        simple_slope = beta['ave_cos_BERT_diary_Human'] + beta['interaction'] * z_values
        slope_se = np.sqrt(se['ave_cos_BERT_diary_Human']**2 + (z_values**2 * se['interaction']**2))

        t_values = simple_slope / slope_se
        p_values = 2 * (1 - norm.cdf(np.abs(t_values)))

        # 有意性の範囲をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(z_values, simple_slope, label='Simple Slope')
        plt.fill_between(z_values, simple_slope - 1.96 * slope_se, simple_slope + 1.96 * slope_se, color='gray', alpha=0.2, label='95% CI')
        plt.axhline(0, color='red', linestyle='--', label='Zero Effect')
        plt.xlabel("Moderator (group_c)")
        plt.ylabel("Simple Slope of ave_cos_BERT_diary_Human")
        plt.title(f"Johnson-Neyman Plot for {dependent_var}")
        plt.legend()
        plt.show()

    else:
        print("交互作用は有意ではありません。ジョンソン-ネイマン法をスキップします。\n")

# 各従属変数に対してロバスト回帰を実行
for dependent_var in dependent_variables:
    run_robust_regression_with_vif_f_r2_jn(dependent_var)
