import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDFが格納されているディレクトリ
pdf_directory = "./pdfs"

# ベクトルデータベースを保存するディレクトリ
db_directory = "../vector_metadata"

# 日付を検出するための正規表現パターン（例: "11月11日"）
# date_pattern = re.compile(r"(\d{1,2})月(\d{1,2})日")
date_pattern = re.compile(r"(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日")

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=130,  # チャンクサイズに設定
            chunk_overlap=50,  # チャンクの重複に設定
            length_function=len,
        )

        # 各日記エントリに分割し、データと日付をリストとして保持
        split_documents = text_splitter.split_documents(documents)
        date_matches = date_pattern.findall(documents[0].page_content)  # すべての日付を一度に抽出

        # ベクトルデータベースを初期化
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        student_db_dir = os.path.join(db_directory, student_id)
        os.makedirs(student_db_dir, exist_ok=True)
        database = Chroma(persist_directory=student_db_dir, embedding_function=embeddings)

        # 各日記エントリに対して、メタデータ付きでデータを追加
        for i, doc in enumerate(split_documents):  # すべての日記エントリを処理
            if i < len(date_matches):  # 日付がある場合のみ設定
                month, day = date_matches[i]
                doc.metadata["date"] = f"{month}月{day}日"
                print(f"{month}月{day}日")
            else:
                doc.metadata["date"] = "日付不明"  # 日付がない場合は「日付不明」
            database.add_documents([doc])

print("Vector databases with metadata created successfully.")
