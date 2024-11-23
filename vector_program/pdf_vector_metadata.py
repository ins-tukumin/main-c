import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# OpenAI APIキーを設定
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# PDFが格納されているディレクトリ
pdf_directory = "./pdfs"

# ベクトルデータベースを保存するディレクトリ
db_directory = "../vector_2nd"

# 日記データを日付ごとに分割する関数
def split_entries_by_date(text):
    entries = []
    pattern = r"([0-9０-９]{2}月[0-9０-９]{2}日)"  # 日付のパターンを正規表現で定義
    sections = re.split(pattern, text)
    for i in range(1, len(sections), 2):
        date = sections[i].strip()      # 日付を取得
        content = sections[i + 1].strip()  # 日記の内容を取得
        if content:
            # 日付と内容を結合して1つのエントリとして追加
            entries.append(f"{date} {content}")
    return entries

# 各PDFファイルに対してベクトルデータベースを作成
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        student_id = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(pdf_directory, pdf_file)

        # PDFを開いてテキストを抽出
        try:
            loader = PyMuPDFLoader(file_path=pdf_path)
            documents = loader.load()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            continue

        # 全テキストを取得し、日付ごとにエントリを分割
        full_text = " ".join([doc.page_content for doc in documents])
        entries = split_entries_by_date(full_text)

        # Embeddingsとデータベースのセットアップ
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )

        student_db_dir = os.path.join(db_directory, student_id)
        if not os.path.exists(student_db_dir):
            os.makedirs(student_db_dir)

        database = Chroma(
            persist_directory=student_db_dir,
            embedding_function=embeddings,
        )

        # 各エントリを `Document` オブジェクトに変換して、`page_content` に格納
        data = [Document(page_content=entry) for entry in entries]
        # 日付ごとに分割された内容をベクトルデータベースに追加
        # data = [{"page_content": entry} for entry in entries]
        database.add_documents(data)

print("Vector databases created successfully.")
