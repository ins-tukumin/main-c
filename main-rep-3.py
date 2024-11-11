import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from firebase_admin import credentials, firestore
import firebase_admin
import datetime
import pytz

# 現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.initge = []
if "past" not in st.session_state:
    st.session_state.past = []

# クエリパラメータからユーザーIDを取得
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True

# 会話テンプレートに`chat_history`を追加
template = """
    今日の出来事を振り返って、自由に感想を語ってもらいましょう。以下の日記情報と会話履歴が参考です。
    日記の日付を考慮しつつ、ユーザーの話を自然に促進してください。
    敬語を使わず、友達のような口調で話してください。
    ------
    過去の日記:
    {context}
    ------
    会話履歴:
    {chat_history}
    ------
    {question}
    Answer:
    """
prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

select_model = "gpt-4"
select_temperature = 0.5

# Firebaseの設定
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

    db_path = f"./vector/{user_id}"
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Chroma データベースをセットアップ
        database = Chroma(persist_directory=db_path, embedding_function=embeddings)

        chat = ChatOpenAI(model=select_model, temperature=select_temperature)
        retriever = database.as_retriever()

        # ConversationBufferMemoryを初期化
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory = st.session_state.memory
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "generated" not in st.session_state:
            st.session_state.generated = []
        if "past" not in st.session_state:
            st.session_state.past = []
        if "count" not in st.session_state:
            st.session_state.count = 0

        # 文脈と日付を組み合わせる関数
        def format_context_with_date(retrieved_docs):
            context_with_dates = []
            for doc in retrieved_docs:
                content = doc.page_content
                date_info = doc.metadata.get("date", "不明な日付")
                context_with_dates.append(f"【日付: {date_info}】\n{content}")
            return "\n\n".join(context_with_dates)

        # 入力を基に応答を取得する
        def on_input_change():
            st.session_state.count += 1
            user_message = st.session_state.user_message
            with st.spinner("相手からの返信を待っています..."):
                retrieved_docs = retriever.get_relevant_documents(user_message)
                context_with_dates = format_context_with_date(retrieved_docs)
                response = chain({
                    "question": user_message,
                    "context": context_with_dates,
                    "chat_history": memory.chat_memory  # 会話履歴を追加
                })
                response_text = response["answer"]
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({"Human": user_message, "AI": response_text})

        # 会話の表示
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
                message(st.session_state.generated[i], key=str(i) + "keyg", avatar_style="micah")

        # 入力欄
        with st.container():
            if st.session_state.count >= 5:
                group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_5cZeI9RbaCdozTU"
                group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
                st.markdown(f'会話は終了です。アンケート回答はこちら: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
            else:
                st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
                st.button("送信", on_click=on_input_change)
    else:
        st.error(f"No vector database found for student ID {user_id}.")
