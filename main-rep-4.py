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

#現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.initge = []
if "past" not in st.session_state:
    st.session_state.past = []

# クエリパラメータからユーザーIDを取得
#query_params = st.query_params
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]
is_second = 'second' in query_params
# user_id = int(user_id)

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True




# これは予備実験のプロンプト！（一部削除、、、）エピソード記憶を参照して話題に出すというのは、例えば、「あの日に遊園地に行ったときに乗ったあのアトラクションで感じたあの感情は今の感情に似ているね」といった具合です。
template = """
# 役割と目的
- あなたはユーザーの一日を振り返るためのAIアシスタントです。
- ユーザーに対して会話を促し、適度に問いかけを行いながら、自由に感想を話してもらうことを目指します。
- あなたの会話には、ユーザーの日記に基づいたエピソード記憶を適切に盛り込み、ユーザーの経験を反映させた返答を行ってください。

# 日記の使用方法
- あなたは、ユーザーの日記情報を読んだ上で、彼/彼女の状況を深く理解しながら会話を進めます。
- 日記に書かれている内容を参考にし、エピソード記憶を自然に会話に織り交ぜてください。
- ただし、「日記を読んだ」といった動作を示す表現（例: “あなたの日記を読んでみると”）は、会話の中に含めないでください。

# エピソード記憶について
- エピソード記憶とは、ユーザーの過去の個人的な経験や出来事を記憶するものです。
- 例えば、特定の出来事を参照し、「あの日の体験に似ているね」といった形で過去の感情や経験を関連付けて会話を進めてください。

# 出力ルール
- 100字以内で返答を行ってください。
- 敬語は使わず、友達として親しみやすい口調で会話を進めてください。
- 必要に応じて、ユーザーの日記から情報を引用し、文脈に合わせた返答を生成してください。

# 入力情報
## ユーザーが事前に書いた日記:
- <ctx>{context}</ctx>

## チャット履歴:
- <hs>{chat_history}</hs>

## ユーザーの質問:
- {question}

# 出力
Answer:
"""

# 会話のテンプレートを作成
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template,
)
select_model = "gpt-4o"
select_temperature = 0.5
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if user_id:
    if not firebase_admin._apps:
        # 環境変数を読み込む
        type = st.secrets["type"]
        project_id = st.secrets["project_id"]
        private_key_id = st.secrets["private_key_id"]
        private_key = st.secrets["private_key"].replace('\\n', '\n')
        client_email = st.secrets["client_email"]
        client_id = st.secrets["client_id"]
        auth_uri = st.secrets["auth_uri"]
        token_uri = st.secrets["token_uri"]
        auth_provider_x509_cert_url = st.secrets["auth_provider_x509_cert_url"]
        client_x509_cert_url = st.secrets["client_x509_cert_url"]
        universe_domain = st.secrets["universe_domain"]
        # Firebase認証情報を設定
        cred = credentials.Certificate({
            "type": type,
            "project_id": project_id,
            "private_key_id": private_key_id,
            "private_key": private_key,
            "client_email": client_email,
            "client_id": client_id,
            "auth_uri": auth_uri,
            "token_uri": token_uri,
            "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
            "client_x509_cert_url": client_x509_cert_url,
            "universe_domain": universe_domain
            })
        default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    db_path = f"./vector_lab_databases/{user_id}"
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )

        database = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,  # エンベディング関数を提供
        )

        chat = ChatOpenAI(
            model=select_model,
            temperature=select_temperature,
        )

        retriever = database.as_retriever()

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )

        memory = st.session_state.memory

        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        #--------------------------------------------
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
                response_text = response["answer"]
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""
            #st.session_state["user_message"] = ""
            Agent_1_Human_Agent = "Human" 
            Agent_2_AI_Agent = "AI" 
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({
                Agent_1_Human_Agent: user_message,
                Agent_2_AI_Agent: response_text
            })
        # 会話履歴を表示するためのスペースを確保
        chat_placeholder = st.empty()
        # 会話履歴を表示
        with chat_placeholder.container():
            message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
                key_generated = str(i) + "keyg"
                message(st.session_state.generated[i], key=str(key_generated), avatar_style="micah")
                
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


