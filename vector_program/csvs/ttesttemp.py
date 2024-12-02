from scipy.stats import t
import numpy as np

# データ入力
n1, mean1, std1 = 48, 0.5561, 0.0747
n2, mean2, std2 = 45, 0.5940, 0.0829

# t統計量の計算
t_stat = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))

# 自由度の計算
df = ((std1**2 / n1) + (std2**2 / n2))**2 / (
    ((std1**2 / n1)**2 / (n1 - 1)) + ((std2**2 / n2)**2 / (n2 - 1))
)

# 両側検定のp値を計算
p_value = 2 * (1 - t.cdf(abs(t_stat), df))

# 結果表示
print(f"t-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df:.4f}")
print(f"p-value: {p_value:.4f}")

# 有意性の判断
alpha = 0.05  # 有意水準
if p_value < alpha:
    print("結論: 帰無仮説を棄却する (p < 0.05)")
else:
    print("結論: 帰無仮説を棄却できない (p >= 0.05)")
