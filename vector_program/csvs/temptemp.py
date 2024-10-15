import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import statistics
import csv
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Firebase初期化 (JSON形式の認証情報を使用)
cred_dict = {
    "type":"service_account",
    "project_id":"main-r",
    "private_key_id":"ffc5a139288747153f225cc9cf2381c334cda3c3",
    "private_key":"-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCeOorVdNEX2Ryu\n/p3mhtjSZPavbJYhJNCnL60LZ6dukR4ghbzUZV6KuobKwL1wwKO8EJnUM5U1x15A\n3dvnm2uJOnijCmkmpkmPNalBWPV8i31nVxecPWk6hzByJ/oxLInzjBHy/GuZTILr\nVlzccMSoGCGwmpLbxeK1cLj+L65Obi3XmEeCdrV3o7BEowjB8q9mwfe7mq6Asyp3\ndakZXH7MXZCX3PpXh2MFM7SCE1zhgiuE8lQyJ2XfvoFx07zSiy+E16hHvTS77+H5\nxHmirHD5jcUd1tsAxe+rGqKlpEWj/xGR/gRPTsBuXXfrixUS1tS+nFabFhkF5jKG\nt8HMSfCBAgMBAAECggEAB4T7tRI0n6ADQ3O5OEvfQPxoVsTfy5puygsze4/E/rWp\ny1zfsO5SVGBx6G/JcPLOtTDKxrFe3WvomEeS3EpgDWj4c7MadSpA46vmq8LZA76z\nSnZymku21P+4tywcMTsPIUJeVgBV8raWoC1+A0Herqha7YQjB4u7buj4+ad1bqet\nlxoLynQo14nCQYBeyOYCYRgoyDaVFYpqxY9LbUveOIxoH5Aw4uo/2hOJAIOwWyMY\n0pk2jN44Br1IpOVJi+aTMdOPjV8r/9MHCXux5zbQoPbzN3BN8AyaeTVKpSLvu8+T\nkPT57d13/g4W5c8LSGXCbvqLA1cT+5btX1/lG2yxMwKBgQDapVa4LZknFdMl93EC\n77l0Oz0IZCXXhUeG63pX8b3Pgr2U/Bhl8lXhI3k5OHFv/KvMNLaZf2d4wCpg9Ucm\ncrc1te3b0FAVmHuTTbPdVGPSz5Q2eI+P0SIqlfgaaS42npkz9OA152mTZQDwkGxk\nZcSCyqitvfyqBWj7HHnBvutqGwKBgQC5Qs6zFjxtZ7l1xa9EJRgf1MyUPNYDJwaD\nCkVLb0VLgh/TK6BgWzyKcclaSCMl/lJzZPGeAU6gpKt67b0ri4FcXpy/7+hNnukp\nG4uA0ZL6KNvUEcaVqvjCQySZYUdCNX+P5P5VobABt5xRHWU0+cHSpw8nqS9aeu1/\n4yF6yTQ5kwKBgEHbY09+jI71R/A7o9KammWkIjIQ2EUeY/kDnIo4yk9ite/WLxMl\n9zAlGzJdCe4NUUHk7ss1UNlSKHGj75ZpHz4SWl7HVBftIeuwj+iurpKk66OslLFg\n8MWa/mwWGlFhXAwGSjJyTZ6T4cCT/9INxS8QE9ahTyV3E7PvU81D0GzZAoGAf59J\nO0+3IvsQZNRg78XJ/6udnwTlvVg2ATGjGNs3VlP2zodAPQC9DPZj6kDFjdfMPtgs\nJlfqLXoi81UxOv0oiVRYEVUYp9gv8PSbvoshABoDje0M62/TXCfa35qG91JZZOww\nVRdEY3p0QeDJJpxjbFVPeFfxWhhS4gW6u5Y91ucCgYA+bT25d+AObddpp7rdWiIc\nl0MHk1RP/5rjCe9M9DAN/VKEqWI21vmRPW/UflrDL68pVl1/PEOHitWra6/RD4h4\njhEMOI/8iZl1QFLWzdkXyYwKDGjnb7eCay95W3irX7N6fXR57VpaELwFPcVmB3rP\nfd+xPDRntusV1G+P49HCHg==\n-----END PRIVATE KEY-----\n",
    "client_email":"firebase-adminsdk-m1jdu@main-r.iam.gserviceaccount.com",
    "client_id":"114270809957158975243",
    "auth_uri":"https://accounts.google.com/o/oauth2/auth",
    "token_uri":"https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-m1jdu%40main-r.iam.gserviceaccount.com",
    "universe_domain":"googleapis.com"
}
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 形態素解析を行い、指定された品詞（名詞、動詞、形容詞）を抽出する関数
def mecab_tokenize_with_filter(text, filter_pos=None):
    if filter_pos is None:
        filter_pos = ['名詞', '動詞', '形容詞']
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(text)

    words = []
    while node:
        pos = node.feature.split(",")[0]
        if pos in filter_pos:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

# 各条件ごとの形態素解析と品詞フィルタリングを行い、1つのテキストにまとめる
def prepare_texts(texts, filter_pos=None):
    return " ".join([mecab_tokenize_with_filter(text, filter_pos) for text in texts])

# コサイン類似度を計算する関数
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# CSVファイル読み込み
conversation_csv_file = "../../group_assignment.csv"
diary_csv_file = "mainfiles/mainfiles/dialy_file.csv"
df_conversation = pd.read_csv(conversation_csv_file)
df_diary = pd.read_csv(diary_csv_file)

# group列がgroupcの行だけを抽出し、user_id列をリストにする
user_ids = df_conversation[df_conversation['group'] == 'groupb']['user_id'].tolist()

# Firestoreに存在しないuser_idを確認
non_existent_user_ids = []

# 集計する日付リスト
dates = ["2024-10-02", "2024-10-03", "2024-10-04"]

# CSV出力
with open('conversation_log_analysis_groupb_cos_diary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダー行
    writer.writerow([
        'userid', 'day1_AI', 'day1_Human', 'day1_cos_AI_Human', 'day1_cos_diary_AI', 'day1_cos_diary_Human',
        'day2_AI', 'day2_Human', 'day2_cos_AI_Human', 'day2_cos_diary_AI', 'day2_cos_diary_Human',
        'day3_AI', 'day3_Human', 'day3_cos_AI_Human', 'day3_cos_diary_AI', 'day3_cos_diary_Human',
        'AI_ave', 'Human_ave', 'ave_cos_AI_Human', 'ave_cos_diary_AI', 'ave_cos_diary_Human'
    ])

    # Firestoreに存在するuser_idに対して処理
    for user_id in user_ids:
        user_ref = db.collection(str(user_id))
        if not user_ref.get():
            non_existent_user_ids.append(user_id)
            continue

        # 日記データのフィルタリング
        user_diary_data = df_diary[df_diary['user_id'] == user_id]

        # データを保存する辞書
        conversation_data = {date: {'AI': [], 'Human': [], 'cos_AI_Human': None, 'cos_diary_AI': None, 'cos_diary_Human': None} for date in dates}

        # 各date_strごとにAIとHumanのテキストを取得
        for date_str in dates:
            docs = user_ref.order_by("__name__").start_at([date_str]).end_at([date_str + "\uf8ff"]).get()
            
            ai_texts = []
            human_texts = []

            for doc in docs:
                data = doc.to_dict()
                if '2.AI' in data:
                    ai_text = data['2.AI']
                    ai_texts.append(ai_text)
                    conversation_data[date_str]['AI'].append(len(ai_text))
                if '1.Human' in data:
                    human_text = data['1.Human']
                    human_texts.append(human_text)
                    conversation_data[date_str]['Human'].append(len(human_text))
            
            # コサイン類似度の計算
            if ai_texts and human_texts:
                ai_text_combined = prepare_texts(ai_texts)
                human_text_combined = prepare_texts(human_texts)
                conversation_data[date_str]['cos_AI_Human'] = calculate_cosine_similarity(ai_text_combined, human_text_combined)

            # 日記データとAI/Humanの類似度計算
            diary_texts = user_diary_data[user_diary_data['StartDate'] < date_str]['Q1'].tolist()  # 前日までの日記データ
            if diary_texts:
                diary_text_combined = prepare_texts(diary_texts)
                if ai_texts:
                    conversation_data[date_str]['cos_diary_AI'] = calculate_cosine_similarity(diary_text_combined, ai_text_combined)
                if human_texts:
                    conversation_data[date_str]['cos_diary_Human'] = calculate_cosine_similarity(diary_text_combined, human_text_combined)

        # 平均計算
        result = {"userid": user_id}
        cos_values_AI_Human = []
        cos_values_diary_AI = []
        cos_values_diary_Human = []

        for i, date in enumerate(dates):
            ai_lengths = conversation_data[date]['AI']
            human_lengths = conversation_data[date]['Human']
            cos_sim_AI_Human = conversation_data[date]['cos_AI_Human']
            cos_sim_diary_AI = conversation_data[date]['cos_diary_AI']
            cos_sim_diary_Human = conversation_data[date]['cos_diary_Human']

            result[f'day{i+1}_AI'] = statistics.mean(ai_lengths) if ai_lengths else 0
            result[f'day{i+1}_Human'] = statistics.mean(human_lengths) if human_lengths else 0
            result[f'day{i+1}_cos_AI_Human'] = cos_sim_AI_Human if cos_sim_AI_Human is not None else 0
            result[f'day{i+1}_cos_diary_AI'] = cos_sim_diary_AI if cos_sim_diary_AI is not None else 0
            result[f'day{i+1}_cos_diary_Human'] = cos_sim_diary_Human if cos_sim_diary_Human is not None else 0

            if cos_sim_AI_Human is not None:
                cos_values_AI_Human.append(cos_sim_AI_Human)
            if cos_sim_diary_AI is not None:
                cos_values_diary_AI.append(cos_sim_diary_AI)
            if cos_sim_diary_Human is not None:
                cos_values_diary_Human.append(cos_sim_diary_Human)

        # 日付範囲全体の全体平均
        all_ai_lengths = [length for date in dates for length in conversation_data[date]['AI']]
        all_human_lengths = [length for date in dates for length in conversation_data[date]['Human']]
        result['AI_ave'] = statistics.mean(all_ai_lengths) if all_ai_lengths else 0
        result['Human_ave'] = statistics.mean(all_human_lengths) if all_human_lengths else 0
        result['ave_cos_AI_Human'] = statistics.mean(cos_values_AI_Human) if cos_values_AI_Human else 0
        result['ave_cos_diary_AI'] = statistics.mean(cos_values_diary_AI) if cos_values_diary_AI else 0
        result['ave_cos_diary_Human'] = statistics.mean(cos_values_diary_Human) if cos_values_diary_Human else 0

        # CSVに書き込む
        writer.writerow([
            result['userid'],
            result[f'day1_AI'], result[f'day1_Human'], result[f'day1_cos_AI_Human'], result[f'day1_cos_diary_AI'], result[f'day1_cos_diary_Human'],
            result[f'day2_AI'], result[f'day2_Human'], result[f'day2_cos_AI_Human'], result[f'day2_cos_diary_AI'], result[f'day2_cos_diary_Human'],
            result[f'day3_AI'], result[f'day3_Human'], result[f'day3_cos_AI_Human'], result[f'day3_cos_diary_AI'], result[f'day3_cos_diary_Human'],
            result['AI_ave'], result['Human_ave'], result['ave_cos_AI_Human'], result['ave_cos_diary_AI'], result['ave_cos_diary_Human']
        ])

print("CSVファイルへの出力が完了しました")

# Firestoreに存在しなかったuser_idを表示
if non_existent_user_ids:
    print(f"Firestoreに存在しなかったuser_id: {non_existent_user_ids}")
