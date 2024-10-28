from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# 1. モデルのロード（日本語モデルを使用）
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# 2. サンプルテキスト（プロトコル分析に使用する発話や文章）
texts = [
    "今日は新しいプロジェクトを始めた。",
    "問題が発生したので、解決方法を考えている。",
    "どうしたらチーム全体が効率的に働けるだろう？",
    "感情をコントロールするのは難しい。",
    "明日までにレポートを書かなければならない。",
    "チームメンバーにフィードバックを伝えるべきか悩んでいる。",
    "自分の役割が果たせていないように感じる。"
]

# 3. テキストをSBERTでエンコード（ベクトルに変換）
embeddings = model.encode(texts)

# 4. クラスタリングの実行（カテゴリ数は3に設定）
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
labels = kmeans.labels_

# 5. 分類結果を表示
for i, label in enumerate(labels):
    print(f"Text: {texts[i]} => Cluster: {label}")

# 6. クラスタごとの内容をまとめて表示
from collections import defaultdict

clustered_texts = defaultdict(list)
for i, label in enumerate(labels):
    clustered_texts[label].append(texts[i])

print("\nClustered Protocol Analysis:")
for cluster, texts in clustered_texts.items():
    print(f"\nCluster {cluster}:")
    for text in texts:
        print(f"  - {text}")
