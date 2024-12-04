import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr

# CSVファイルの読み込み
file_path = "BIGBERT.csv"  # ここにCSVファイルのパスを指定
df = pd.read_csv(file_path)

# Remove group 'groupb'
df = df[df['group'] != 'groupb']

# 相関を計算する列を指定
column_x = "ave_cos_BERT_diary_Human"  # 1つ目の列の名前を指定
column_y = "stan_topic_count"  # 2つ目の列の名前を指定

# NaNを含む行を削除（相関分析に必要）
data = df[[column_x, column_y]].dropna()

# 正規性検定（Shapiro-Wilk検定）
shapiro_x_stat, shapiro_x_p = shapiro(data[column_x])
shapiro_y_stat, shapiro_y_p = shapiro(data[column_y])

print(f"Shapiro-Wilk Test for {column_x}: Statistic={shapiro_x_stat:.3f}, p-value={shapiro_x_p:.3e}")
print(f"Shapiro-Wilk Test for {column_y}: Statistic={shapiro_y_stat:.3f}, p-value={shapiro_y_p:.3e}")

# ピアソン相関係数とp値
pearson_corr, pearson_p = pearsonr(data[column_x], data[column_y])
print(f"Pearson correlation coefficient: {pearson_corr:.3f}, p-value: {pearson_p:.3e}")

# スピアマン順位相関係数とp値
spearman_corr, spearman_p = spearmanr(data[column_x], data[column_y])
print(f"Spearman rank correlation: {spearman_corr:.3f}, p-value: {spearman_p:.3e}")

# 解釈
if pearson_p < 0.05:
    print("ピアソン相関: 有意な相関があります。")
else:
    print("ピアソン相関: 有意な相関は見られません。")

if spearman_p < 0.05:
    print("スピアマン順位相関: 有意な相関があります。")
else:
    print("スピアマン順位相関: 有意な相関は見られません。")
