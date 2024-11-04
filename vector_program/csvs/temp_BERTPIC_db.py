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
# 認証情報を直接JSON形式で埋め込む
cred_dict = {
    #json
}
# Firebase初期化 (JSON形式の認証情報を使用)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
# Firestoreクライアントを作成
db = firestore.client()

# CSVファイルの読み込みとデータの抽出
def load_texts_from_csv(file_path, user_id):
    # CSVファイルを読み込み
    df = pd.read_csv(file_path)
    # 指定のuser_idでフィルタリングし、Q1列のデータをリストに格納
    texts = df[df['user_id'] == user_id]['Q1'].dropna().tolist()
    return texts

# 指定されたファイルパスとuser_idを使用してデータを取得
file_path = 'mainfiles/mainfiles/dialy_file.csv'  # ファイルのパスを指定
user_id = '2141268'  # 対象のuser_idを指定
texts = load_texts_from_csv(file_path, user_id)

# 1. モデルのロード（日本語対応のSBERT）
sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# 2. HDBSCANのパラメータを調整
hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=2, prediction_data=True)
umap_model = UMAP(n_neighbors=20, n_components=5, metric='cosine')

# 3. BERTopicの初期化（UMAPを有効化）
topic_model = BERTopic(
    embedding_model=sbert_model,
    hdbscan_model=hdbscan_model,
    umap_model=umap_model
)

# 5. 文ごとに「。」で区切ってリスト化
processed_texts = []
for text in texts:
    processed_texts.extend([sentence for sentence in text.split("。") if sentence])  # 空白を避ける

# 6. トピックの抽出と割り当て
topics, _ = topic_model.fit_transform(processed_texts)

# 7. トピックごとの文章を収集
topic_texts = defaultdict(list)
for text, topic in zip(processed_texts, topics):
    if topic != -1:  # トピック -1 はスキップ
        topic_texts[topic].append(text)

# 8. GPT-4を使用してトピック名を自動生成
topic_names = {}
for topic, texts in topic_texts.items():
    prompt = "以下の文章の共通トピックを一言で表してください:\n" + "\n".join(texts)
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "あなたは専門的な文章を理解して簡潔なトピック名を提供するアシスタントです。日本語で話します。"},
            {"role": "user", "content": prompt}
        ]
    )
    topic_name = response.choices[0].message.content.strip()
    topic_names[topic] = topic_name

print("\nトピックごとの文章:")
for i, (text, topic) in enumerate(zip(processed_texts, topics)):
    print(f"文 {i}: トピック {topic} -> {text}")

# トピックの名前と出現頻度を表示
print("トピックの名前と出現頻度:")
topic_counts = Counter(topics)
for topic, name in topic_names.items():
    print(f"トピック {topic}: {name} - {topic_counts[topic]} 回")

# 最も頻出の5つのトピックを抽出
topic_counts = Counter(topics)
if -1 in topic_counts:
    del topic_counts[-1]
top_5_topics = topic_counts.most_common(5)
top_5_topic_ids = [topic for topic, _ in top_5_topics]

# 出現頻度上位5つのトピックとその頻度
print("トップ5のトピック:", top_5_topics)

# Firestoreから指定のuser_idの"2.AI"ログを取得し、リストに格納
def extract_ai_logs_from_firestore(user_id):
    ai_texts = []
    collection_ref = db.collection(user_id)
    docs = collection_ref.order_by("__name__").get()

    for doc in docs:
        data = doc.to_dict()
        if 'Human' in data:
            ai_texts.extend([sentence for sentence in data['Human'].split("。") if sentence])  # 「。」で区切ってリスト化
    
    return ai_texts

# user_idを指定してAIログを抽出
ai_texts = extract_ai_logs_from_firestore(user_id)
print("抽出されたAIテキスト:", ai_texts)

# トピックの割り当てとカウント
topic_occurrences = Counter()
total_char_counts = Counter()

# GPTを使って各文に最も適切なトピックを割り当て
for text in ai_texts:
    # プロンプトを作成
    prompt = f"以下の文章に最も適切なトピックを一つ選んでください:\n\n文章: {text}\n\nトピックの候補: {', '.join(topic_names.values())}"
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "あなたはテキストの内容に基づいて最も適切なトピックを選ぶアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
    )

    assigned_topic_name = response.choices[0].message.content.strip()

    # トピック名に対応するトピックIDを検索
    assigned_topic_id = None
    for topic_id, topic_name in topic_names.items():
        if assigned_topic_name == topic_name:
            assigned_topic_id = topic_id
            break

    # トピックが見つかった場合、カウンタを進める
    if assigned_topic_id is not None:
        topic_occurrences[assigned_topic_id] += 1
        total_char_counts[assigned_topic_id] += len(text)

    # デバッグ用出力
    print(f"テキスト: {text} -> 割り当てられたトピック: {assigned_topic_id}")

# トピックごとの登場回数と総文字数を表示
print("トピックごとの登場回数と総文字数:")
for topic_id, count in topic_occurrences.items():
    topic_name = topic_names.get(topic_id, "不明なトピック")
    print(f"トピック {topic_id} ({topic_name}): {count} 回, 総文字数: {total_char_counts[topic_id]}")