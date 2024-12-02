from scipy.stats import t
import numpy as np

def students_t_test(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    # プールされた標準偏差の計算
    sp = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # t統計量の計算
    t_stat = (mean1 - mean2) / (sp * np.sqrt(1/n1 + 1/n2))
    
    # 自由度の計算
    df = n1 + n2 - 2
    
    # 両側検定のp値を計算
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    # 結果表示
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.4f}")
    
    # 有意性の判断
    if p_value < alpha:
        print(f"結論: 帰無仮説を棄却します（p < {alpha}）")
    else:
        print(f"結論: 帰無仮説を棄却できません（p >= {alpha}）")
    
    return t_stat, df, p_value


# データ入力
n1, mean1, std1 = 48,0.1737 , 0.0651 
n2, mean2, std2 = 45, 0.2276  ,0.0783 

# t検定の実行
students_t_test(mean1, std1, n1, mean2, std2, n2)
