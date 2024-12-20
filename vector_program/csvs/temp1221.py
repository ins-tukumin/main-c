from scipy.stats import ttest_ind
import numpy as np

# 各変数のデータ
data = {
    "PANAS_P": {"mean_a": 2.597, "sem_a": 0.113, "n_a": 30, 
                "mean_c": 2.954, "sem_c": 0.097, "n_c": 28},
    "PANAS_N": {"mean_a": 1.741, "sem_a": 0.095, "n_a": 30, 
                "mean_c": 2.061, "sem_c": 0.124, "n_c": 28},
    "competence": {"mean_a": 4.016, "sem_a": 0.095, "n_a": 30, 
                   "mean_c": 3.643, "sem_c": 0.124, "n_c": 28},
    "warmth": {"mean_a": 4.504, "sem_a": 0.061, "n_a": 30, 
               "mean_c": 4.631, "sem_c": 0.051, "n_c": 28},
    "satisfaction": {"mean_a": 4.000, "sem_a": 0.113, "n_a": 30, 
                     "mean_c": 3.679, "sem_c": 0.130, "n_c": 28},
    "effectiveness": {"mean_a": 4.105, "sem_a": 0.096, "n_a": 30, 
                      "mean_c": 3.631, "sem_c": 0.117, "n_c": 28},
    "efficiency": {"mean_a": 4.302, "sem_a": 0.092, "n_a": 30, 
                   "mean_c": 3.964, "sem_c": 0.117, "n_c": 28},
    "willingness": {"mean_a": 3.953, "sem_a": 0.114, "n_a": 30, 
                    "mean_c": 3.548, "sem_c": 0.133, "n_c": 28},
    "understanding": {"mean_a": 4.047, "sem_a": 0.107, "n_a": 30, 
                      "mean_c": 3.869, "sem_c": 0.126, "n_c": 28}
}

# t検定結果を保存するリスト
results = []

# 各変数についてt検定を実行
for variable, values in data.items():
    mean_a, sem_a, n_a = values["mean_a"], values["sem_a"], values["n_a"]
    mean_c, sem_c, n_c = values["mean_c"], values["sem_c"], values["n_c"]
    
    # 標準偏差の計算
    sd_a = sem_a * np.sqrt(n_a)
    sd_c = sem_c * np.sqrt(n_c)
    
    # ダミーデータの生成
    data_a = np.random.normal(loc=mean_a, scale=sd_a, size=n_a)
    data_c = np.random.normal(loc=mean_c, scale=sd_c, size=n_c)
    
    # t検定の実施
    t_stat, p_value = ttest_ind(data_a, data_c, equal_var=False)
    results.append({"variable": variable, "t_stat": t_stat, "p_value": p_value})

# 結果の表示
print("t検定の結果:")
for result in results:
    print(f"変数: {result['variable']}, t値: {result['t_stat']:.2f}, p値: {result['p_value']:.4f}")
