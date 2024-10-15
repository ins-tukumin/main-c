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
    a
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
