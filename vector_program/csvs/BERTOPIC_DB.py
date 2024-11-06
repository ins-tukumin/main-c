import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import openai
from umap import UMAP
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

# Firebaseの初期化（既に初期化されている場合はこの部分をスキップ）
cred_dict = {
    
}
# Firebase初期化 (JSON形式の認証情報を使用)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
# Firestoreクライアントを作成
db = firestore.client()

# 日記データの読み込み（diary_file.csv の Q1 列）
def load_diary_texts(file_path, user_id):
    #print(f"Loading diary texts for user_id: {user_id} from {file_path}")
    df = pd.read_csv(file_path)
    user_diary_texts = df[df['user_id'] == user_id]['Q1'].dropna().tolist()
    #print(f"Loaded {len(user_diary_texts)} diary entries for user_id: {user_id}")
    return user_diary_texts

# Firestoreから指定のuser_idの会話ログを取得
def extract_ai_logs_from_firestore(user_id):
    #print(f"Fetching conversation logs for user_id: {user_id}")
    ai_texts = []
    total_length = 0  # 総文字数
    sentence_count = 0  # 文の数

    collection_ref = db.collection(user_id)
    docs = collection_ref.order_by("__name__").get()

    for doc in docs:
        data = doc.to_dict()
        if 'Human' in data:
            sentences = [sentence for sentence in data['Human'].split("。") if sentence]
            ai_texts.extend(sentences)
            total_length += sum(len(sentence) for sentence in sentences)
            sentence_count += len(sentences)

    #print(f"Fetched {len(ai_texts)} sentences from conversation logs.")
    #print(f"Total characters: {total_length}, Total sentences: {sentence_count}")
    return ai_texts, total_length, sentence_count

# 日記データをトピック分けしてトピック名をつける
def generate_topic_model(processed_texts):
    #print("Generating topics from diary texts...")
    sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=2, prediction_data=True)
    umap_model = UMAP(n_neighbors=20, n_components=5, metric='cosine')
    topic_model = BERTopic(embedding_model=sbert_model, hdbscan_model=hdbscan_model, umap_model=umap_model)

    # トピック抽出と割り当て
    topics, _ = topic_model.fit_transform(processed_texts)
    topic_texts = defaultdict(list)
    for text, topic in zip(processed_texts, topics):
        if topic != -1:
            topic_texts[topic].append(text)

    #print(f"Extracted {len(topic_texts)} topics.")

    # トピック名生成
    topic_names = {}
    existing_topic_names = []
    for topic, texts in topic_texts.items():
        #print(f"\nGenerating name for topic {topic} with {len(texts)} sentences:")
        #print("Sample text in this topic:", texts[:3])  # 最初の3文を表示
        prompt = (
            "以下の文章のユニークな共通トピックを、一言で表してください。"
            "ただし、次のリストにある名前と似た名前は避けてください:\n"
            + ", ".join(existing_topic_names) + "\n\n"
            + "文章:\n" + "\n".join(texts)
        )
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "あなたは専門的な文章を理解して簡潔で、他と重複しないトピック名を提供するアシスタントです。日本語で話します。"},
                {"role": "user", "content": prompt}
            ]
        )
        topic_name = response.choices[0].message.content.strip()
        topic_names[topic] = topic_name
        existing_topic_names.append(topic_name)
        #print(f"Assigned name '{topic_name}' to topic {topic}")

    print("\nトピックごとの文章:")
    for i, (text, topic) in enumerate(zip(processed_texts, topics)):
        print(f"文 {i}: トピック {topic} -> {text}")

    # トピックの名前と出現頻度を表示
    print("トピックの名前と出現頻度:")
    topic_counts = Counter(topics)
    for topic, name in topic_names.items():
        print(f"トピック {topic}: {name} - {topic_counts[topic]} 回")

    return topic_model, topic_names

# 会話ログをトピックに分類
def classify_conversation_logs(user_id, topic_model, topic_names):
    #print(f"Classifying conversation logs for user_id: {user_id}")
    ai_texts, total_length, sentence_count = extract_ai_logs_from_firestore(user_id)
    
    topic_occurrences = Counter()
    total_char_counts = Counter()
    none_count = 0

    for i, text in enumerate(ai_texts):
        prompt = f"以下の文章に最も適切なトピックを一つ選んでください。適切なトピックがない場合は'None'と回答してください。Noneにも興味があるので、無理にトピックを選ぶ必要はないです。:\n\n文章: {text}\n\nトピック候補: {', '.join(topic_names.values())}"
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "あなたはテキストの内容に基づいて最も適切なトピックを選ぶアシスタントです。"},
                {"role": "user", "content": prompt}
            ]
        )
        assigned_topic_name = response.choices[0].message.content.strip()

        # トピック名からIDを取得
        assigned_topic_id = next((tid for tid, name in topic_names.items() if name == assigned_topic_name), None)
        
        if assigned_topic_id is not None:
            topic_occurrences[assigned_topic_id] += 1
            total_char_counts[assigned_topic_id] += len(text)
            #print(f"Text {i + 1}: '{text}' -> Assigned to topic '{assigned_topic_name}' (ID: {assigned_topic_id})")
        else:
            none_count += 1
            #print(f"Text {i + 1}: '{text}' -> No suitable topic found, classified as 'None'.")

        # デバッグ用出力
        print(f"テキスト: {text} -> 割り当てられたトピック: {assigned_topic_id}")

    #print(f"Total classified texts: {sum(topic_occurrences.values())}, None count: {none_count}")
    # トピックごとの登場回数と総文字数を表示
    print("トピックごとの登場回数と総文字数:")
    for topic_id, count in topic_occurrences.items():
        topic_name = topic_names.get(topic_id, "不明なトピック")
        print(f"トピック {topic_id} ({topic_name}): {count} 回, 総文字数: {total_char_counts[topic_id]}")
    print(f"None: {none_count} 回")


    return topic_occurrences, total_char_counts, none_count, total_length, sentence_count

# CSVに出力
def save_to_csv(user_id, topic_occurrences, total_char_counts, none_count, total_length, sentence_count):
    total_topic_count = sum(topic_occurrences.values())
    output_data = {
        "user_id": [user_id],
        "total_length": [total_length],
        "sentence_count": [sentence_count],
        "topic_count": [total_topic_count],
        "none_count": [none_count]
    }
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(f"{user_id}_protocol.csv", index=False, encoding="utf-8")
    #print(f"Data saved to '{user_id}_protocol.csv'.")

# 実行処理
user_id = '3954203'  # 指定したユーザーID
file_path = 'mainfiles/analysis/dialy_file.csv'

# 日記データの読み込み
diary_texts = load_diary_texts(file_path, user_id)
processed_texts = [sentence for text in diary_texts for sentence in text.split("。") if sentence]
#print(f"\nProcessing user_id {user_id} with {len(processed_texts)} processed sentences.")

# トピック生成と名前付け
topic_model, topic_names = generate_topic_model(processed_texts)

# 会話ログを分類
topic_occurrences, total_char_counts, none_count, total_length, sentence_count = classify_conversation_logs(user_id, topic_model, topic_names)

# CSVに保存
save_to_csv(user_id, topic_occurrences, total_char_counts, none_count, total_length, sentence_count)
