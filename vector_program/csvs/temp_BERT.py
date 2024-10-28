import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict

# 1. モデルのロード（日本語SBERTモデル）
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# 2. サンプルテキスト
texts = [
    "今日は雨。それなりの雨量だ。明日も雨なので、今日は散歩には行けそうもない。今日もクラウドワークスのプロジェクト案件で受注した記事を朝から書いている、寝るまで書き続ける感じになりそう。受注があることは有り難い。週末は晴れそうなので、気分転換は週末にしよう。",
    "今朝は、上の子の昨日の体育祭の話を聞いて、とても楽しかったことを知った。お弁当もとても喜んでくれて良かった。頑張って作って良かったと思う。とても暑かったが熱中症にならず安心した。下の子はテスト前の勉強漬けで疲れているが、ペースをつかんできたようだ。私も微力ながら勉強のアドバイスをしたりサポートをしている。",
    "今日は自主学習でエクセルについて勉強した。その後昨日少し取り掛かっていたエクセルの関数を使う宿題に取り掛かったが、全くやり方がわからずあれこれと調べものをしているうちに時間がたってしまった。もう少し基本からゆっくりやり直さなければならない。午前中ちょっと買い物するつもりが、近所の野菜などの産地直売店など寄っているうちに大幅に一日の予算を過ぎてしまった。",
    "今日は体調もかなり回復して、全く普通に仕事ができたのでホッとした。元気一杯というわけではないが、ほぼ平常運転に近い感じで過ごせた１日だった。食事もきちんと摂れた。天気予報通りに気温が一気に下がったが、思ったほどは寒くは感じず、まだまだ半袖で過ごしても大丈夫だと思った。とりあえず週の真ん中を越したので、今週もなんとか乗り切れそうだ。",
    "今日は午前中晴れていたけど、午後からは曇っていた。昨日より少し暑くて27度あったらしい。 今日も犬の散歩を済ませ、朝ごはん兼お昼ご飯を食べて、お風呂掃除とかした後テレビを観ながらコーヒーとおやつを頂いた。 再放送していたドラマ「古畑任三郎」が今日で終わってしまった。最終回は昨日だったけど、今日はイチロー選手がゲストの回のスペシャルだった。第一シリーズから第三シリーズまで再放送してくれたから嬉しかったけど、他のスペシャルも放送してほしかったなぁ…。 寂しくて古畑ロスになってしまいそう…。",
    "今日は新作のゲームの発売日だ",
    "FPSは数あるゲームタイトルでも最高のカテゴリーだ"
]

# 3. テキストをSBERTでエンコード（ベクトルに変換）
embeddings = model.encode(texts)

# 4. シルエット分析で最適なクラスタ数を決定する関数
def find_optimal_clusters(embeddings, max_clusters=10):
    best_score = -1  # 初期スコア
    best_k = 2  # シルエットスコアの最小クラスタ数

    # クラスタ数の範囲を調整 (データ数以下に制限)
    for k in range(2, min(max_clusters + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

# 5. エルボー法を使うバックアップ関数
def elbow_method(embeddings, max_clusters=10):
    sse = []
    cluster_range = range(1, min(max_clusters + 1, len(embeddings)))

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    # SSEをプロット
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# 6. クラスタリングの実行
try:
    # シルエット分析で最適なクラスタ数を取得
    optimal_k = find_optimal_clusters(embeddings)
    print(f"\nOptimal number of clusters: {optimal_k}")

    # 最適なクラスタ数でKMeansを実行
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # クラスタごとの内容を表示
    clustered_texts = defaultdict(list)
    for i, label in enumerate(labels):
        clustered_texts[label].append(texts[i])

    print("\nClustered Protocol Analysis:")
    for cluster, cluster_texts in clustered_texts.items():
        print(f"\nCluster {cluster}:")
        for text in cluster_texts:
            print(f"  - {text}")

except ValueError as e:
    print(f"Error: {e}")
    print("Fallback to Elbow Method.")
    # エルボー法をバックアップとして実行
    elbow_method(embeddings)
