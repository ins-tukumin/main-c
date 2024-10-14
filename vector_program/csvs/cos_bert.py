from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===== ステップ1: モデルの準備 =====
model_sbert = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')  # Sentence-BERTの多言語対応モデルを使用

# ===== ステップ2: 複数の文を含む日記のリストを準備 =====
diary1 = [
    "昨夜も次男が夜泣きを何度もしたけど、私自身心に余裕があったためか、落ち着いて対応できたと思う。朝は5時か6時に起きることが日常になったが、夫が毎回早く起きて面倒を見てくれるのでありがたいと思う。長男は朝眠そうだったが、大好きなフルーツがあると聞いてにこにこして起きてきたのでとてもかわいかった。",
    "昨夜も次男の夜泣きが続いたので、途中から夫に託して私は寝てしまった。そのおかげで私はまとまって眠れたので良かった。朝は涼しくて窓を開けたら気持ちのいい風が入ってきてよかった。でもしばらくするとたくさん雨が降ってきてしまった。 長男が帰りにスニーカーがべたべたになっているかなあと心配している。",
    "昨夜は次男の覚醒時間が長かったので、立ってあやしながら寝ていた。子育てをしてからこんな器用なこともできるようになった。我ながらよくやっていると感じる。30分くらいあやしたら寝たので、私も寝ると少しまとめて寝てくれた。今日は長男がパンが食べたいと言ったので、チーズを乗せて焼いたらとても喜んでいた。いつもご飯を食べるのが遅いのに、大好きなチーズはぺろっとたいらげてしまって、子供らしくてかわいいなと思った。"
]

diary2 = [
    "a",
    "arfte",
    "t5t"
    ]

# ===== ステップ3: Sentence-BERT を使って日記をベクトル化 =====
def get_mean_sentence_embedding(text_list, model):
    """複数の文を含むリストを1つのベクトルにまとめる"""
    embeddings = model.encode(text_list)  # 各文をベクトル化
    return np.mean(embeddings, axis=0)    # 平均ベクトルを返す

# 各日記の埋め込みベクトルを取得
embedding1 = get_mean_sentence_embedding(diary1, model_sbert)
embedding2 = get_mean_sentence_embedding(diary2, model_sbert)

# ===== ステップ4: コサイン類似度を計算 =====
cos_sim_sbert = cosine_similarity([embedding1], [embedding2])
print(f"日記1 と 日記2 の Sentence-BERT コサイン類似度: {cos_sim_sbert[0][0]}")
