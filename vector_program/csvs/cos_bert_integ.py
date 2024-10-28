import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import statistics
import csv
import pandas as pd
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# ===== Firebase初期化 (JSON形式の認証情報を使用) =====
# cred_dict = {}
cred = credentials.Certificate("mainfiles/main-r.json")
# cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===== Sentence-BERTモデルの準備 =====
model_sbert = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# ===== 形態素解析を行い、指定された品詞を抽出する関数 =====
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

# ===== 複数のテキストをまとめて形態素解析する関数 =====
def prepare_texts(texts, filter_pos=None):
    return " ".join([mecab_tokenize_with_filter(text, filter_pos) for text in texts])

# ===== Sentence-BERTでのコサイン類似度計算関数 =====
def get_mean_sentence_embedding(text_list, model):
    embeddings = model.encode(text_list)
    return np.mean(embeddings, axis=0)

def calculate_cosine_similarity_sbert(text_list1, text_list2, model):
    embedding1 = get_mean_sentence_embedding(text_list1, model)
    embedding2 = get_mean_sentence_embedding(text_list2, model)
    return cosine_similarity([embedding1], [embedding2])[0][0]

# ===== CSVファイルの読み込み =====
conversation_csv_file = "../../group_assignment.csv"
diary_csv_file = "mainfiles/analysis/dialy_file.csv"
df_conversation = pd.read_csv(conversation_csv_file)
df_diary = pd.read_csv(diary_csv_file)

# ===== group列がgroupbの行のuser_idを取得 =====
user_ids = df_conversation[df_conversation['group'] == 'groupb']['user_id'].tolist()

# ===== Firestoreに存在しないuser_idを確認 =====
non_existent_user_ids = []

# ===== 集計する日付リスト =====
dates = ["2024-10-02", "2024-10-03", "2024-10-04"]

# ===== CSV出力 =====
with open('BERT_cos_b.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'userid', 'day1_cos_AI_Human', 'day1_cos_diary_AI', 'day1_cos_diary_Human',
        'day2_cos_AI_Human', 'day2_cos_diary_AI', 'day2_cos_diary_Human',
        'day3_cos_AI_Human', 'day3_cos_diary_AI', 'day3_cos_diary_Human',
         'ave_cos_AI_Human', 'ave_cos_diary_AI', 'ave_cos_diary_Human'
    ])

    # Firestoreに存在するuser_idごとに処理
    for user_id in user_ids:
        user_ref = db.collection(str(user_id))
        if not user_ref.get():
            non_existent_user_ids.append(user_id)
            continue

        # ユーザーの日記データ取得
        user_diary_data = df_diary[df_diary['user_id'] == user_id]
        conversation_data = {date: {'AI': [], 'Human': [], 'cos_AI_Human': None, 'cos_diary_AI': None, 'cos_diary_Human': None} for date in dates}

        # 各日付ごとのAI/Humanテキストを取得
        for date_str in dates:
            docs = user_ref.order_by("__name__").start_at([date_str]).end_at([date_str + "\uf8ff"]).get()
            ai_texts, human_texts = [], []

            for doc in docs:
                data = doc.to_dict()
                if '2.AI' in data:
                    ai_texts.append(data['2.AI'])
                if '1.Human' in data:
                    human_texts.append(data['1.Human'])

            # AIとHumanのコサイン類似度計算
            if ai_texts and human_texts:
                conversation_data[date_str]['cos_AI_Human'] = calculate_cosine_similarity_sbert(
                    ai_texts, human_texts, model_sbert
                )

            # 日記とAI/Humanの類似度計算
            diary_texts = user_diary_data[user_diary_data['StartDate'] < date_str]['Q1'].tolist()
            if diary_texts:
                if ai_texts:
                    conversation_data[date_str]['cos_diary_AI'] = calculate_cosine_similarity_sbert(
                        diary_texts, ai_texts, model_sbert
                    )
                if human_texts:
                    conversation_data[date_str]['cos_diary_Human'] = calculate_cosine_similarity_sbert(
                        diary_texts, human_texts, model_sbert
                    )
        # ===== 結果を集計してCSVに書き込み =====
        result = {"userid": user_id}

        # 各日付の類似度の集計と丸め
        cos_AI_Human_values = []
        cos_diary_AI_values = []
        cos_diary_Human_values = []

        for i, date in enumerate(dates):
            # 各日付ごとの値を取得し、存在しない場合は0を代入
            cos_AI_Human = conversation_data[date]['cos_AI_Human'] or 0
            cos_diary_AI = conversation_data[date]['cos_diary_AI'] or 0
            cos_diary_Human = conversation_data[date]['cos_diary_Human'] or 0

            # 値を丸めて格納
            result[f'day{i+1}_cos_AI_Human'] = round(cos_AI_Human, 3)
            result[f'day{i+1}_cos_diary_AI'] = round(cos_diary_AI, 3)
            result[f'day{i+1}_cos_diary_Human'] = round(cos_diary_Human, 3)

            # 各日の値をリストに追加（平均計算用）
            cos_AI_Human_values.append(cos_AI_Human)
            cos_diary_AI_values.append(cos_diary_AI)
            cos_diary_Human_values.append(cos_diary_Human)

        # 平均値を計算して丸める
        result['ave_cos_AI_Human'] = round(statistics.mean(cos_AI_Human_values), 3) if cos_AI_Human_values else 0
        result['ave_cos_diary_AI'] = round(statistics.mean(cos_diary_AI_values), 3) if cos_diary_AI_values else 0
        result['ave_cos_diary_Human'] = round(statistics.mean(cos_diary_Human_values), 3) if cos_diary_Human_values else 0

        # ===== CSVに書き込み =====
        writer.writerow([
            result['userid'],
            *[result.get(f'day{i+1}_cos_AI_Human', 0) for i in range(3)],
            *[result.get(f'day{i+1}_cos_diary_AI', 0) for i in range(3)],
            *[result.get(f'day{i+1}_cos_diary_Human', 0) for i in range(3)],
            result['ave_cos_AI_Human'], result['ave_cos_diary_AI'], result['ave_cos_diary_Human']
        ])

print("CSVファイルへの出力が完了しました")

# Firestoreに存在しなかったuser_idを表示
if non_existent_user_ids:
    print(f"Firestoreに存在しなかったuser_id: {non_existent_user_ids}")
