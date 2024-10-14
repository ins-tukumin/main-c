import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import statistics
import csv
import pandas as pd

# 認証情報を直接JSON形式で埋め込む
cred_dict = {
 
}

# Firebase初期化 (JSON形式の認証情報を使用)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# CSVファイル読み込み
csv_file = "../../group_assignment.csv"  # 読み込みたいCSVファイルのパスを指定
df = pd.read_csv(csv_file)

# group列がgroupaの行だけを抽出し、user_id列をリストにする
user_ids = df[df['group'] == 'groupc']['user_id'].tolist()

# Firestoreに存在しないuser_idを確認
non_existent_user_ids = []


# 集計する日付リスト (例)
dates = ["2024-10-02", "2024-10-03", "2024-10-04"]

# データを保存する辞書
conversation_data = {date: {'AI': [], 'Human': []} for date in dates}

# まとめてCSV出力するためにファイルを一度開く
with open('mainfiles/analysis/conversation_log_analysis_gourpc.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダー行
    writer.writerow(['userid', 'day1_AI', 'day1_Human', 'day2_AI', 'day2_Human', 'day3_AI', 'day3_Human', 'AI_ave', 'Human_ave'])

    # Firestoreに存在するuser_idに対して処理
    for user_id in user_ids:
        user_ref = db.collection(str(user_id))  # user_idを文字列として扱う
        if not user_ref.get():  # ドキュメントが存在しない場合
            non_existent_user_ids.append(user_id)
            continue  # このuser_idについてはスキップして次へ進む
        
        # データを保存する辞書
        conversation_data = {date: {'AI': [], 'Human': []} for date in dates}

        # ドキュメントIDを日付ごとに取得
        for date_str in dates:
            # FirebaseクエリでドキュメントIDがdate_strを含むものを取得
            docs = user_ref.order_by("__name__").start_at([date_str]).end_at([date_str + "\uf8ff"]).get()
            
            for doc in docs:
                data = doc.to_dict()
                # データがAI と Human を含んでいるか確認
                if 'AI' in data:
                    ai_text = data['AI']
                    conversation_data[date_str]['AI'].append(len(ai_text))

                if 'Human' in data:
                    human_text = data['Human']
                    conversation_data[date_str]['Human'].append(len(human_text))

        # 平均計算
        result = {"userid": user_id}
        for i, date in enumerate(dates):
            ai_lengths = conversation_data[date]['AI']
            human_lengths = conversation_data[date]['Human']
            result[f'day{i+1}_AI'] = statistics.mean(ai_lengths) if ai_lengths else 0
            result[f'day{i+1}_Human'] = statistics.mean(human_lengths) if human_lengths else 0

        # 日付範囲全体の全体平均
        all_ai_lengths = [length for date in dates for length in conversation_data[date]['AI']]
        all_human_lengths = [length for date in dates for length in conversation_data[date]['Human']]

        result['AI_ave'] = statistics.mean(all_ai_lengths) if all_ai_lengths else 0
        result['Human_ave'] = statistics.mean(all_human_lengths) if all_human_lengths else 0

        # 各user_idごとのデータをまとめて1つのCSVに書き込む
        writer.writerow([
            result['userid'],
            result[f'day1_AI'],
            result[f'day1_Human'],
            result[f'day2_AI'],
            result[f'day2_Human'],
            result[f'day3_AI'],
            result[f'day3_Human'],
            result['AI_ave'],
            result['Human_ave']
        ])

print("CSVファイルへの出力が完了しました")

# Firestoreに存在しなかったuser_idを表示
if non_existent_user_ids:
    print(f"Firestoreに存在しなかったuser_id: {non_existent_user_ids}")