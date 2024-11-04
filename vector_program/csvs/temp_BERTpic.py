import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import openai  # OpenAI APIの使用
from umap import UMAP

# CSVファイルの読み込みとデータの抽出
def load_texts_from_csv(file_path, user_id):
    # CSVファイルを読み込み
    df = pd.read_csv(file_path)
    
    # 指定のuser_idでフィルタリングし、Q1列のデータをリストに格納
    texts = df[df['user_id'] == user_id]['Q1'].dropna().tolist()
    
    return texts

# 指定されたファイルパスとuser_idを使用してデータを取得
file_path = 'mainfiles/mainfiles/dialy_file.csv'  # ファイルのパスを指定
user_id = '5024777'  # 対象のuser_idを指定
texts = load_texts_from_csv(file_path, user_id)

# 1. モデルのロード（日本語対応のSBERT）
sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# 2. HDBSCANのパラメータを調整（少量のデータに対応）
hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=2)

umap_model = UMAP(n_neighbors=20, n_components=5, metric='cosine')

# 3. BERTopicの初期化（UMAPを無効化）
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

# 9. トピックごとの名前と出現頻度を表示
print("トピックの名前と出現頻度:")
topic_counts = Counter(topics)
for topic, name in topic_names.items():
    print(f"トピック {topic}: {name} - {topic_counts[topic]} 回")

print("\nトピックごとの文章:")
for i, (text, topic) in enumerate(zip(processed_texts, topics)):
    print(f"文 {i}: トピック {topic} -> {text}")

# 最も頻出の5つのトピックを抽出
topic_counts = Counter(topics)
# -1 を除外
if -1 in topic_counts:
    del topic_counts[-1]
top_5_topics = topic_counts.most_common(5)
top_5_topic_ids = [topic for topic, _ in top_5_topics]

# 出現頻度上位5つのトピックとその頻度
print("トップ5のトピック:", top_5_topics)
