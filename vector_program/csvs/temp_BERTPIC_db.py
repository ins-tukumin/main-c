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
    print(f"Loading diary texts for user_id: {user_id} from {file_path}")
    df = pd.read_csv(file_path)
    user_diary_texts = df[df['user_id'] == user_id]['Q1'].dropna().tolist()
    print(f"Loaded {len(user_diary_texts)} diary entries for user_id: {user_id}")
    return user_diary_texts

# Firestoreから指定のuser_idの会話ログを取得
def extract_ai_logs_from_firestore(user_id):
    print(f"Fetching conversation logs for user_id: {user_id}")
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

    print(f"Fetched {len(ai_texts)} sentences from conversation logs.")
    print(f"Total characters: {total_length}, Total sentences: {sentence_count}")
    return ai_texts, total_length, sentence_count

# 日記データをトピック分けしてトピック名をつける
def generate_topic_model(processed_texts):
    print("Generating topics from diary texts...")
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

    print(f"Extracted {len(topic_texts)} topics.")

    # トピック名生成
    topic_names = {}
    existing_topic_names = []
    for topic, texts in topic_texts.items():
        print(f"\nGenerating name for topic {topic} with {len(texts)} sentences:")
        print("Sample text in this topic:", texts[:3])  # 最初の3文を表示
        prompt = (
            "以下の文章の共通トピックを、一言で表してください。"
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
        print(f"Assigned name '{topic_name}' to topic {topic}")

    return topic_model, topic_names

# 会話ログをトピックに分類
def classify_conversation_logs(user_id, topic_model, topic_names):
    print(f"Classifying conversation logs for user_id: {user_id}")
    ai_texts, total_length, sentence_count = extract_ai_logs_from_firestore(user_id)
    
    topic_occurrences = Counter()
    total_char_counts = Counter()
    none_count = 0

    for i, text in enumerate(ai_texts):
        prompt = f"以下の文章に最も適切なトピックを一つ選んでください。適切なトピックがない場合は'None'と回答してください。Noneにも興味があるので、無理にトピックを選ぶ必要はないです。:\n\n文章: {text}\n\nトピック候補: {', '.join(topic_names.values())}"
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
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
            print(f"Text {i + 1}: '{text}' -> Assigned to topic '{assigned_topic_name}' (ID: {assigned_topic_id})")
        else:
            none_count += 1
            print(f"Text {i + 1}: '{text}' -> No suitable topic found, classified as 'None'.")

    print(f"Total classified texts: {sum(topic_occurrences.values())}, None count: {none_count}")
    return topic_occurrences, total_char_counts, none_count, total_length, sentence_count

# すべてのユーザーの結果をまとめてCSVに出力
def save_all_to_csv(all_results):
    output_df = pd.DataFrame(all_results)
    output_df.to_csv("groupc_protocol.csv", index=False, encoding="utf-8")
    print("All data saved to 'groupc_protocol.csv'.")

# 実行処理
file_path = 'mainfiles/analysis/dialy_file.csv'
df = pd.read_csv(file_path)

# `group` 列が `groupc` の `user_id` を対象に処理
user_ids = df[df['group'] == 'groupc']['user_id'].unique()
all_results = []  # 全ユーザーの結果を格納するリスト

for user_id in user_ids:
    print(f"\nProcessing user_id {user_id}...")

    # 日記データの読み込み
    diary_texts = load_diary_texts(file_path, user_id)
    processed_texts = [sentence for text in diary_texts for sentence in text.split("。") if sentence]
    print(f"Total processed sentences for user_id {user_id}: {len(processed_texts)}")

    # トピック生成と名前付け
    topic_model, topic_names = generate_topic_model(processed_texts)

    # 会話ログを分類
    topic_occurrences, total_char_counts, none_count, total_length, sentence_count = classify_conversation_logs(user_id, topic_model, topic_names)

    # 各ユーザーの処理結果をリストに追加
    all_results.append({
        "user_id": user_id,
        "total_length": total_length,
        "sentence_count": sentence_count,
        "topic_count": sum(topic_occurrences.values()),
        "none_count": none_count
    })

# まとめてCSVに保存
save_all_to_csv(all_results)
