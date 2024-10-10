# 必要なライブラリのインストールとインポート
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import BertTokenizer
import MeCab
from sklearn.feature_extraction.text import CountVectorizer

# MeCabの初期化
mecab = MeCab.Tagger()

# データの読み込み
df = pd.read_csv("mainfiles/1001_selected_file.csv")
texts = df['Q1'].tolist()

# 名詞と動詞のみを抽出する前処理関数を定義
def extract_nouns_and_verbs(text):
    parsed = mecab.parse(text)
    words = []
    for line in parsed.splitlines():
        if line == "EOS":
            break
        cols = line.split("\t")
        if len(cols) >= 4:
            word, pos = cols[0], cols[3]
            if "名詞" in pos or "動詞" in pos:
                words.append(word)
    return " ".join(words)

# 前処理を行い、名詞・動詞のみを抽出したテキストリストを作成
texts = [extract_nouns_and_verbs(text) for text in texts]
texts = [text for text in texts if text.strip() != ""]  # 空の文書を削除

# BERT用トークナイザーでトリミング
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
MAX_TOKEN_LENGTH = 511
texts = [" ".join(tokenizer.tokenize(text)[:MAX_TOKEN_LENGTH]) for text in texts]

# 日本語BERTで埋め込みを作成
sentence_model = SentenceTransformer("cl-tohoku/bert-base-japanese")
embeddings = sentence_model.encode(texts, show_progress_bar=True)

# ベクトル化のパラメータ調整
vectorizer_model = CountVectorizer(min_df=1, max_df=0.9)

# BERTopicでトピックモデリング
topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=5)
topics, probabilities = topic_model.fit_transform(texts, embeddings)

# 結果を確認
print("トピック情報:", topic_model.get_topic_info())
