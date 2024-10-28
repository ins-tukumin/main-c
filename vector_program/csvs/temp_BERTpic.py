from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# 1. モデルのロード（日本語対応のSBERT）
sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# 2. HDBSCANのパラメータを調整（少量のデータに対応）
hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)

# 3. BERTopicの初期化（UMAPを無効化）
topic_model = BERTopic(
    embedding_model=sbert_model,  # SBERTを使用
    hdbscan_model=hdbscan_model,  # HDBSCANの設定
    umap_model=None  # 少量データ向けにUMAPを無効化
)

# 4. サンプルテキスト（日本語の文章）
texts = [
    "今日は雨。それなりの雨量だ。明日も雨なので、今日は散歩には行けそうもない。今日もクラウドワークスのプロジェクト案件で受注した記事を朝から書いている、寝るまで書き続ける感じになりそう。受注があることは有り難い。週末は晴れそうなので、気分転換は週末にしよう。",
    "今朝は、上の子の昨日の体育祭の話を聞いて、とても楽しかったことを知った。お弁当もとても喜んでくれて良かった。頑張って作って良かったと思う。とても暑かったが熱中症にならず安心した。下の子はテスト前の勉強漬けで疲れているが、ペースをつかんできたようだ。私も微力ながら勉強のアドバイスをしたりサポートをしている。",
    "今日は自主学習でエクセルについて勉強した。その後昨日少し取り掛かっていたエクセルの関数を使う宿題に取り掛かったが、全くやり方がわからずあれこれと調べものをしているうちに時間がたってしまった。もう少し基本からゆっくりやり直さなければならない。",
    "今日は体調もかなり回復して、全く普通に仕事ができたのでホッとした。元気一杯というわけではないが、ほぼ平常運転に近い感じで過ごせた１日だった。",
    "今日は午前中晴れていたけど、午後からは曇っていた。昨日より少し暑くて27度あったらしい。 今日も犬の散歩を済ませ、朝ごはん兼お昼ご飯を食べて、お風呂掃除とかした後テレビを観ながらコーヒーとおやつを頂いた。",
    "今日は新作のゲームの発売日だ",
    "FPSは数あるゲームタイトルでも最高のカテゴリーだ"
]

# 5. トピックの抽出と割り当て
topics, _ = topic_model.fit_transform(texts)

# 6. トピックごとの文章を表示
print("\nトピックごとの文章:")
for i, (text, topic) in enumerate(zip(texts, topics)):
    print(f"文 {i}: トピック {topic} -> {text}")

# 7. トピックの出現頻度をカウントして表示
from collections import Counter

topic_counts = Counter(topics)
print("\nトピックの出現頻度:")
for topic, count in topic_counts.items():
    print(f"トピック {topic}: {count} 回")

# 8. トピックごとのキーワードを表示
print("\nトピックの詳細:")
for topic, words in topic_model.get_topic_info()[['Topic', 'Name']].values:
    print(f"トピック {topic}: {words}")
