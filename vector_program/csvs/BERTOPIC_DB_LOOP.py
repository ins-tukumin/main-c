import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import openai
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
import json

# Firebaseの初期化（既に初期化されている場合はこの部分をスキップ）
firebase_json_path = "mainfiles/main-c.json"  # 認証情報JSONファイルのパス

# JSONファイルから認証情報を読み込む
with open(firebase_json_path, "r", encoding="utf-8") as json_file:
    cred_dict = json.load(json_file)

# Firebase初期化 (JSON形式の認証情報を使用)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 日記データの読み込み（CSVのQ1列を抽出）
def load_diary_texts(df, user_id):
    """
    ユーザーIDに対応する日記データを抽出する関数。

    :param df: Pandas DataFrame (日記データ)
    :param user_id: 対象のユーザーID
    :return: 該当ユーザーの日記テキストのリスト
    """
    user_diary_texts = df[df['user_id'] == user_id]['Q1'].dropna().tolist()
    return user_diary_texts

# Firestoreから指定した日付リストでHuman会話ログを取得
def extract_ai_logs_from_firestore(user_id, dates):
    """
    Firestore から指定した日付リストの範囲で Human 会話ログを抽出する関数。

    :param user_id: Firestore のユーザー ID
    :param dates: 日付リスト（例: ["2024-10-02", "2024-10-03"]）
    :return: Human テキストのリスト、総文字数、文数
    """
    print(f"Fetching Human conversation logs for user_id: {user_id} in date range: {dates}")
    human_texts = []
    total_length = 0  # 総文字数
    sentence_count = 0  # 文の数

    # Firestore コレクションリファレンスを取得
    collection_ref = db.collection(user_id)

    # 日付ごとにログを取得
    for date_str in dates:
        try:
            # 日付に基づいてクエリを構築
            docs = collection_ref.order_by("__name__").start_at([date_str]).end_at([date_str + "\uf8ff"]).get()
            for doc in docs:
                data = doc.to_dict()
                if 'Human' in data:
                    # 文を分割してリストに追加
                    sentences = [sentence for sentence in data['Human'].split("。") if sentence]
                    human_texts.extend(sentences)
                    # 総文字数と文数を更新
                    total_length += sum(len(sentence) for sentence in sentences)
                    sentence_count += len(sentences)

        except Exception as e:
            print(f"Error fetching logs for date {date_str}: {e}")

    print(f"Fetched {len(human_texts)} sentences from Human conversation logs.")
    print(f"Total characters: {total_length}, Total sentences: {sentence_count}")
    return human_texts, total_length, sentence_count

# 日記データをトピック分けしてトピック名を生成
def generate_topic_model(processed_texts):
    sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=2, prediction_data=False)
    topic_model = BERTopic(embedding_model=sbert_model, hdbscan_model=hdbscan_model)

    # トピック抽出と割り当て
    topics, _ = topic_model.fit_transform(processed_texts)
    topic_texts = defaultdict(list)
    for text, topic in zip(processed_texts, topics):
        if topic != -1:
            topic_texts[topic].append(text)

    # トピック名生成
    topic_names = {}
    existing_topic_names = []
    for topic, texts in topic_texts.items():
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

    return topic_model, topic_names

# 会話ログをトピックに分類
def classify_conversation_logs(user_id, topic_model, topic_names, dates):
    ai_texts, total_length, sentence_count = extract_ai_logs_from_firestore(user_id, dates)

    topic_occurrences = Counter()
    total_char_counts = Counter()
    none_count = 0

    for text in ai_texts:
        prompt = f"以下の文章に最も適切なトピックを一つ選んでください。適切なトピックがない場合は'None'と回答してください:\n\n文章: {text}\n\nトピック候補: {', '.join(topic_names.values())}"
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "あなたはテキストの内容に基づいて最も適切なトピックを選ぶアシスタントです。"},
                {"role": "user", "content": prompt}
            ]
        )
        assigned_topic_name = response.choices[0].message.content.strip()
        assigned_topic_id = next((tid for tid, name in topic_names.items() if name == assigned_topic_name), None)

        if assigned_topic_id is not None:
            topic_occurrences[assigned_topic_id] += 1
            total_char_counts[assigned_topic_id] += len(text)
        else:
            none_count += 1

    return topic_occurrences, total_char_counts, none_count, total_length, sentence_count

# 統合CSVに保存
#def save_to_csv(results, output_file):
#    output_df = pd.DataFrame(results)
#    output_df.to_csv(output_file, index=False, encoding="utf-8")
#    print(f"Data saved to '{output_file}'.")

# 統合CSVに保存
def save_to_csv(results, output_file):
    # total_topic列を計算して追加
    for result in results:
        total_topic = result["topic_count"] + result["none_count"]
        result["total_topic"] = total_topic
        # 標準化列の計算
        result["stan_topic_count"] = result["topic_count"] / total_topic if total_topic > 0 else 0
        result["stan_none_count"] = result["none_count"] / total_topic if total_topic > 0 else 0

    # データをDataFrameに変換
    output_df = pd.DataFrame(results)

    # CSVに保存
    output_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Data saved to '{output_file}'.")


# 実行処理
def main():
    file_path = 'mainfiles/analysis/FINAL_dialy_file.csv'
    output_file = 'aggregated_protocol.csv'
    target_group = 'groupa'  # 処理対象のグループ
    dates = ["2024-10-02", "2024-10-03", "2024-10-04"]  # 対象日付リスト

    df = pd.read_csv(file_path)
    user_ids = df[df['group'] == target_group]['user_id'].unique()

    results = []  # 統合出力用のリスト

    for user_id in user_ids:
        print(f"\nProcessing user_id: {user_id}")
        diary_texts = load_diary_texts(df, user_id)
        processed_texts = [sentence for text in diary_texts for sentence in text.split("。") if sentence]

        if not processed_texts:
            print(f"No diary texts found for user_id: {user_id}. Skipping.")
            continue

        topic_model, topic_names = generate_topic_model(processed_texts)
        topic_occurrences, total_char_counts, none_count, total_length, sentence_count = classify_conversation_logs(user_id, topic_model, topic_names, dates)

        results.append({
            "user_id": user_id,
            "total_length": total_length,
            "sentence_count": sentence_count,
            "topic_count": sum(topic_occurrences.values()),
            "none_count": none_count
        })

    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()
