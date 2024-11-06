import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('BIGBERT.csv')

# プロットしたい列のリスト
specified_columns = ['ave_PANAS_P', 'ave_PANAS_N', 'ave_competence','ave_warmth', 'ave_willingness','ave_understanding']  # プロットしたい列名をリストで指定

# 各指定列に対して箱ひげ図を作成
for column in specified_columns:
    plt.figure(figsize=(10, 6))
    boxplot = df.boxplot(column=column, by='group', grid=False)

    # 箱ひげ図のデータを取得
    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        
        # 四分位数とIQRを計算
        Q1 = group_data[column].quantile(0.25)
        Q3 = group_data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # 外れ値のカットオフを計算
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 外れ値のデータをフィルタリング
        outliers = group_data[(group_data[column] < lower_bound) | (group_data[column] > upper_bound)]

        # 外れ値のuser_idを出力
        if not outliers.empty:
            print(f'Outliers for {column} in group {group}:')
            print(outliers['user_id'].tolist())  # 外れ値のuser_idをリストで表示

    # プロットのタイトルとラベルを設定
    plt.title(f'Boxplot of {column} by Group')
    plt.suptitle('')  # 余計なサブタイトルを消す
    plt.xlabel('Group')
    plt.ylabel(column)

    # プロットを表示
    plt.show()
