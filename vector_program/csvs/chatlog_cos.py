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
csv_file = "../../group_assignment.csv"
df = pd.read_csv(csv_file)

# group列がgroupaの行だけを抽出し、user_id列をリストにする
user_ids = df[df['group'] == 'groupc']['user_id'].tolist()

# Firestoreに存在しないuser_idを確認
non_existent_user_ids = []

# 集計する日付リスト
dates = ["2024-10-02", "2024-10-03", "2024-10-04"]

# CSV出力
with open('conversation_log_analysis_groupc_cos.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダー行
    writer.writerow(['userid', 'day1_AI', 'day1_Human', 'day1_cos', 'day2_AI', 'day2_Human', 'day2_cos', 'day3_AI', 'day3_Human', 'day3_cos', 'AI_ave', 'Human_ave', 'ave_cos'])

    # Firestoreに存在するuser_idに対して処理
    for user_id in user_ids:
        user_ref = db.collection(str(user_id))
        if not user_ref.get():
            non_existent_user_ids.append(user_id)
            continue

        # データを保存する辞書
        conversation_data = {date: {'AI': [], 'Human': [], 'cos': None} for date in dates}

        # 各date_strごとにAIとHumanのテキストを取得
        for date_str in dates:
            docs = user_ref.order_by("__name__").start_at([date_str]).end_at([date_str + "\uf8ff"]).get()
            
            ai_texts = []
            human_texts = []

            for doc in docs:
                data = doc.to_dict()
                if 'AI' in data:
                    ai_text = data['AI']
                    ai_texts.append(ai_text)
                    conversation_data[date_str]['AI'].append(len(ai_text))
                if 'Human' in data:
                    human_text = data['Human']
                    human_texts.append(human_text)
                    conversation_data[date_str]['Human'].append(len(human_text))
            
            # コサイン類似度の計算
            if ai_texts and human_texts:
                ai_text_combined = prepare_texts(ai_texts)
                human_text_combined = prepare_texts(human_texts)
                conversation_data[date_str]['cos'] = calculate_cosine_similarity(ai_text_combined, human_text_combined)

        # 平均計算
        result = {"userid": user_id}
        cos_values = []

        for i, date in enumerate(dates):
            ai_lengths = conversation_data[date]['AI']
            human_lengths = conversation_data[date]['Human']
            cos_sim = conversation_data[date]['cos']

            result[f'day{i+1}_AI'] = statistics.mean(ai_lengths) if ai_lengths else 0
            result[f'day{i+1}_Human'] = statistics.mean(human_lengths) if human_lengths else 0
            result[f'day{i+1}_cos'] = cos_sim if cos_sim is not None else 0
            if cos_sim is not None:
                cos_values.append(cos_sim)

        # 日付範囲全体の全体平均
        all_ai_lengths = [length for date in dates for length in conversation_data[date]['AI']]
        all_human_lengths = [length for date in dates for length in conversation_data[date]['Human']]
        result['AI_ave'] = statistics.mean(all_ai_lengths) if all_ai_lengths else 0
        result['Human_ave'] = statistics.mean(all_human_lengths) if all_human_lengths else 0
        result['ave_cos'] = statistics.mean(cos_values) if cos_values else 0

        # CSVに書き込む
        writer.writerow([
            result['userid'],
            result[f'day1_AI'], result[f'day1_Human'], result[f'day1_cos'],
            result[f'day2_AI'], result[f'day2_Human'], result[f'day2_cos'],
            result[f'day3_AI'], result[f'day3_Human'], result[f'day3_cos'],
            result['AI_ave'], result['Human_ave'], result['ave_cos']
        ])

print("CSVファイルへの出力が完了しました")

# Firestoreに存在しなかったuser_idを表示
if non_existent_user_ids:
    print(f"Firestoreに存在しなかったuser_id: {non_existent_user_ids}")
