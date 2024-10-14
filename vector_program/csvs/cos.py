import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# サンプルテキストデータ（条件ごとの文を複数まとめたもの）
texts_condition1 = [
    "今日は大雨の中、梅田に行きました。 美容院でヘアカラーとカットをしてから、西梅田をブラブラと歩きました。 西梅田で最近新しい施設がたくさんオープンしていることはニュースなどを見て知っていましたが、来るのは初めてで、ワクワクしました。 道がわからなくても案内に沿って適当に歩いていると着くので、昔よりもとてもわかりやすく、歩きやすくなったと感じます。街並みも綺麗です。 グラングリーン大阪の芝生に座りたかったですが、雨なので不可能でした。 KITTE大阪に入って、ご当地土産をたくさん見ましたが、結局何も購入しませんでした。 KITTEなので、1階の郵便局で記念に切手を買いました。 大好きな函館の高級回転寿司屋を見つけて驚いたので、思わず入ってしまいました。 つぶ貝と炙り中トロが美味しすぎて幸せでした。 しかし価格が高すぎるので、3皿しか食べず、ビールも飲みませんでした。 ルクアの方に移動し、大阪ステーションシネマに行って、明日観に行く予定の映画の前売りムビチケを購入しました。 ルクアやグランフロントなどもうろうろしたかったですが、子供の下校時間が迫っていることに気付き、慌てて駅に向かいました。 たくさん歩いて靴擦れしてしまいましたが、楽しい一日でした。"
]

texts_condition2 = [
    "昨夜も次男の夜泣きが続いたので、途中から夫に託して私は寝てしまった。そのおかげで私はまとまって眠れたので良かった。朝は涼しくて窓を開けたら気持ちのいい風が入ってきてよかった。でもしばらくするとたくさん雨が降ってきてしまった。 長男が帰りにスニーカーがべたべたになっているかなあと心配している。",
    "昨夜も次男が夜泣きを何度もしたけど、私自身心に余裕があったためか、落ち着いて対応できたと思う。朝は5時か6時に起きることが日常になったが、夫が毎回早く起きて面倒を見てくれるのでありがたいと思う。長男は朝眠そうだったが、大好きなフルーツがあると聞いてにこにこして起きてきたのでとてもかわいかった。"
]

# 形態素解析を行い、指定された品詞（名詞、動詞、形容詞）を抽出する関数
def mecab_tokenize_with_filter(text, filter_pos=None):
    if filter_pos is None:
        filter_pos = ['名詞', '動詞', '形容詞']  # デフォルトで抽出する品詞を指定
    mecab = MeCab.Tagger()  # 形態素解析ツール MeCab の設定（Chasen形式を指定）
    node = mecab.parseToNode(text)
    
    words = []
    while node:
        pos = node.feature.split(",")[0]  # 品詞を取得（例: "名詞", "動詞" など）
        if pos in filter_pos:
            words.append(node.surface)  # 該当する品詞の単語を追加
        node = node.next
    return " ".join(words)

# 各条件ごとの形態素解析と品詞フィルタリングを行い、条件ごとに1つのテキストにまとめる
def prepare_texts(texts, filter_pos=None):
    return " ".join([mecab_tokenize_with_filter(text, filter_pos) for text in texts])

# 条件1と条件2のテキストを品詞フィルタリングし、1つのテキストとして準備
filtered_text1 = prepare_texts(texts_condition1)
filtered_text2 = prepare_texts(texts_condition2)

# フィルタリング後のテキストを表示
print(f"Filtered Text for Condition 1: {filtered_text1}")
print(f"Filtered Text for Condition 2: {filtered_text2}")

# ベクトル化（CountVectorizerを用いて単語頻度ベクトルを作成）
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform([filtered_text1, filtered_text2])

# ベクトルの表示
print(f"Vector Shape: {vectors.shape}")
print(f"Vector for Condition 1: {vectors.toarray()[0]}")
print(f"Vector for Condition 2: {vectors.toarray()[1]}")

# コサイン類似度を計算
cos_sim = cosine_similarity(vectors[0], vectors[1])
print(f"Cosine Similarity between Condition 1 and Condition 2: {cos_sim[0][0]}")
