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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import pytz
import time

# 現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# Streamlitの状態を初期化
if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.initge = []
if "past" not in st.session_state:
    st.session_state.past = []

# クエリパラメータからユーザーIDを取得
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]
is_second = 'second' in query_params

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True

# プロンプトテンプレートに日付情報（entry_date）を追加
template = """
    今日の出来事を振り返って、ユーザーに自由に感想を語ってもらいましょう。適度な問いかけを行って、会話を促進してください。
    私の日記情報も添付します。
    この日記を読んで、私の事をよく理解した上で会話してください。
    必要に応じて、私の日記に書かれている情報を参照して、私の事を理解して会話してください。
    ただ、”あなたの日記を読んでみると”といったような、日記を読んだ動作を直接示すような言葉は出力に含めないでください。
    さらに、この会話では私の日記に含まれる「エピソード記憶」を適切に会話に盛り込んで話してほしいです。エピソード記憶という言葉の意味は以下に示します。
    # エピソード記憶とは、人間の記憶の中でも特に個人的な経験や出来事を覚える記憶の種類の一つです。エピソード記憶は、特定の時間と場所に関連する出来事を含む記憶であり、過去の個人的な経験を詳細に思い出すことができる記憶を指します。
    敬語は使わないでください。私の友達になったつもりで砕けた口調で話してください。
    100字以内で話してください。
    日本語で話してください。
    日記を書いた日：{entry_date}
    日記内容：{context}
    ------
    過去の会話履歴:
    <hs>{chat_history}</hs>
    {question}
    Answer:
"""

# 会話のテンプレートを作成
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "entry_date", "question"],
    template=template,
)

select_model = "gpt-4o"
select_temperature = 0.5

# Streamlitのスタイルを隠す
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

# Firebaseの初期化
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
            "client_x509_cert_url": st.secrets["client_x509_cert_url"],
            "universe_domain": st.secrets["universe_domain"]
        })
        default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()

    db_path = f"./vector_lab_databases/{user_id}"
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        database = Chroma(persist_directory=db_path, embedding_function=embeddings)

        chat = ChatOpenAI(model=select_model, temperature=select_temperature)
        retriever = database.as_retriever()

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

        def on_input_change():
            st.session_state.count += 1
            user_message = st.session_state.user_message
            with st.spinner("相手からの返信を待っています。。。"):
                response = chain({"question": user_message})

                # 検索結果から日記の内容と日付メタデータを取得してプロンプトに追加
                context_data = []
                for doc in response["context"]:
                    entry_date = doc.metadata.get("date", "日付不明")  # 日付メタデータ
                    context_data.append({"content": doc.page_content, "date": entry_date})

                # 検索結果をもとに応答を生成
                full_prompt = ""
                for data in context_data:
                    full_prompt += prompt.format(
                        context=data["content"],
                        entry_date=data["date"],
                        question=user_message,
                        chat_history=memory.load_memory_variables().get("chat_history", "")
                    )

                # ChatOpenAIに応答を生成させる
                response_text = chat(full_prompt)
                
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""

            # Firebase Firestoreに会話履歴を保存
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({
                "Human": user_message,
                "AI": response_text
            })

        # 会話履歴の表示
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
                key_generated = str(i) + "keyg"
                message(st.session_state.generated[i], key=key_generated, avatar_style="micah")

        # 入力フォーム
        with st.container():
            if st.session_state.count >= 5:
                group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_5cZeI9RbaCdozTU"
                group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
                st.markdown(f'これで今回の会話は終了です。こちらをクリックしてアンケートに回答してください。: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
            else:
                user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
                st.button("送信", on_click=on_input_change)
    else:
        st.error(f"No vector database found for student ID {user_id}.")
