import os
import sys
import datetime
import pytz
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# SQLite3 モジュール設定
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Firebase 設定
import firebase_admin
from firebase_admin import credentials, firestore

# 現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# Streamlit の状態を初期化
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []
if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.generated.append(initial_message)
    st.session_state['initialized'] = True

# UIのスタイリングを隠す
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {visibility: hidden; height: 0%; position: fixed;}
                div[data-testid="stDecoration"] {visibility: hidden; height: 0%; position: fixed;}
                div[data-testid="stStatusWidget"] {visibility: hidden; height: 0%; position: fixed;}
                #MainMenu {visibility: hidden; height: 0%;}
                header {visibility: hidden; height: 0%;}
                footer {visibility: hidden; height: 0%;}
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# クエリパラメータからユーザーIDを取得
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]

# プロンプトテンプレート設定
template = """
    今日の出来事を振り返って、ユーザーに自由に感想を語ってもらいましょう。適度な問いかけを行って、会話を促進してください。
    以下は私の日記情報です。この日記を参考にして、私のことを理解した上で会話をしてください。
    ただし、「あなたの日記を読んで」など直接日記を読んだ表現は避けてください。
    ---
    日記情報:
    {context}
    ---
    <hs>{chat_history}</hs>
    質問: {question}
    回答:
"""

# PromptTemplate の設定
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template,
)

# Firebase の初期化
if user_id:
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Vector データベースの読み込み
    db_path = f"./vector_metadata/{user_id}"
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        database = Chroma(persist_directory=db_path, embedding_function=embeddings)
        chat = ChatOpenAI(model="gpt-4o", temperature=0.5)
        retriever = database.as_retriever()

        # メモリ設定
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory = st.session_state.memory

        # ConversationalRetrievalChain の設定
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        # 入力と会話履歴の管理
        def on_input_change():
            user_message = st.session_state.user_message

            # 日記情報を検索してコンテキストを準備
            context_data = []
            retrieved_docs = retriever.get_relevant_documents(user_message)
            for doc in retrieved_docs:
                entry_date = doc.metadata.get("date", "日付不明")
                page_content = doc.page_content
                context_data.append(f"日付: {entry_date}\n内容: {page_content}")

            # コンテキストとして日記情報を組み込む
            full_context = "\n\n".join(context_data)
            full_prompt = template.format(
                chat_history=memory.load_memory_variables().get("chat_history", ""),
                context=full_context,
                question=user_message
            )

            # 応答生成
            response_text = chat(full_prompt)
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""

            # Firebase Firestore に会話履歴を保存
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({"Human": user_message, "AI": response_text})

        # 会話履歴の表示
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for i in range(len(st.session_state.generated)):
                if i == 0:
                    message(st.session_state.generated[i], key="init_greeting", avatar_style="micah")
                else:
                    message(st.session_state.past[i-1], is_user=True, key=f"user_{i}", avatar_style="adventurer", seed="Nala")
                    message(st.session_state.generated[i], key=f"ai_{i}", avatar_style="micah")

        # 入力フォーム
        if "count" not in st.session_state:
            st.session_state.count = 0
        if st.session_state.count < 5:
            user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
            st.button("送信", on_click=on_input_change)
        else:
            st.markdown("これで今回の会話は終了です。")
    else:
        st.error(f"No vector database found for student ID {user_id}.")
