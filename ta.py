import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDFが格納されているディレクトリ
pdf_directory = "./pdfs"

# ベクトルデータベースを保存するディレクトリ
db_directory = "../vector"

# 日付を検出するための正規表現パターン
date_pattern = re.compile(r"(\d{1,2})月(\d{1,2})日")

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
            chunk_size=130,  # 150字程度のチャンクサイズに設定
            chunk_overlap=50,  # 50字程度の重複に設定
            length_function=len,
        )

        data = text_splitter.split_documents(documents)

        # メタデータに日付を追加
        for doc in data:
            # 日付のパターンを検索
            date_match = date_pattern.search(doc.page_content)
            if date_match:
                # 日付情報をメタデータとして保存
                month, day = date_match.groups()
                doc.metadata["date"] = f"{month}月{day}日"

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )

        # 学生IDごとのディレクトリを作成
        student_db_dir = os.path.join(db_directory, student_id)
        if not os.path.exists(student_db_dir):
            os.makedirs(student_db_dir)

        # ベクトルデータベースを作成
        database = Chroma(
            persist_directory=student_db_dir,
            embedding_function=embeddings,
        )

        # メタデータ付きドキュメントを追加
        database.add_documents(data)

print("Vector databases with metadata created successfully.")
