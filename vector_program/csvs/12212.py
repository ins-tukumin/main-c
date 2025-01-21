import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルのパス
input_csv_file = "BIGBERT3rd.csv"

def plot_columns_by_week2_group_with_error(csv_file, weeks, columns, use_se=True):
    """
    week2_groupごとにプロットし、エラーバーとしてSDまたはSEを表示する。

    Parameters:
        csv_file (str): CSVファイルのパス
        weeks (list): プロットする週のラベル
        columns (list): プロットする列名
        use_se (bool): Trueの場合は標準誤差（SE）、Falseの場合は標準偏差（SD）をエラーバーに使用
    """
    # CSVを読み込み
    df = pd.read_csv(csv_file)
    
    # week2_group列のユニークな値を取得
    week2_groups = df['week2_group'].unique()
    
    # プロットの設定
    plt.figure(figsize=(10, 6))

    # 各グループについてプロット
    for week2_group in week2_groups:
        week2_group_data = df[df['week2_group'] == week2_group]  # グループごとのデータを抽出
        
        # 指定された列ごとに平均値とエラーバー（SDまたはSE）を計算
        mean_values = []
        error_values = []
        for column in columns:
            if column in week2_group_data.columns:
                mean_value = week2_group_data[column].mean()
                std_dev = week2_group_data[column].std()
                n = len(week2_group_data[column].dropna())  # サンプルサイズ
                
                # SDまたはSEを計算
                if use_se:
                    error_value = std_dev / (n ** 0.5) if n > 0 else 0  # 標準誤差
                else:
                    error_value = std_dev  # 標準偏差
                
                mean_values.append(mean_value)
                error_values.append(error_value)
            else:
                mean_values.append(None)  # データが存在しない場合
                error_values.append(None)

        # グラフにプロット（エラーバーを含む）
        plt.errorbar(
            weeks,
            mean_values,
            yerr=error_values,
            marker='o',
            # label=f"week2_group: {week2_group}",  # グループ名をラベルに設定
            capsize=5  # エラーバーのキャップのサイズ
        )

    # グラフ設定
    plt.title("competence")
    # plt.xlabel("Weeks")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.rcParams.update({'font.size': 16})
    plt.ylabel("Mean Value")
    #plt.xticks(rotation=45)  # X軸ラベルを45度回転
    plt.ylim(1.0, 5.0)
    plt.yticks(np.arange(1.0, 6.0, 1.0))
    plt.grid(False)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# 実行（例: 複数の列名を指定）
columns_to_plot = ['week2_competence', 'week3_competence', 'week4_competence', 'week5_competence']
weeks = ['week1', 'week2', 'week3', 'week4']

# SDを使用
plot_columns_by_week2_group_with_error(input_csv_file, weeks, columns_to_plot, use_se=False)

# SEを使用
plot_columns_by_week2_group_with_error(input_csv_file, weeks, columns_to_plot, use_se=True)
