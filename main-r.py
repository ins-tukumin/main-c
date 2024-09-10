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
#st.write("クエリパラメータ:", query_params)
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]
is_second = 'second' in query_params
#st.write(f"型: {type(user_id)}") 
user_id = int(user_id)
#st.write(f"こんにちは、{user_id}さん！")


        
#st.write(f"file_path:{file_path}")
#file_path = 'worry2.txt'
    
# 環境変数の読み込み
#from dotenv import load_dotenv
#load_dotenv()
#if 'worries' in st.session_state:
#    st.write(st.session_state.worries)  # デバッグ用
#    if user_id in st.session_state.worries:
#        st.write(f"Worry found for user: {st.session_state.worries[user_id]}")  # デバッグ用
#    else:
#        st.write("User id not found in worries")


if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！。今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
#    st.session_state.past.append("")
#    message(st.session_state.initge[0], key="init_greeting", avatar_style="micah")
    st.session_state['initialized'] = True

# テンプレートの設定
template = """
    敬語は使わないでください。私の友達になったつもりで砕けた口調で話してください。
    100字以内で話してください。
    日本語で話してください。
    私の入力に基づき、次の文脈（<ctx></ctx>で囲まれた部分）とチャット履歴（<hs></hs>で囲まれた部分）を使用して回答してください。:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    {message}
    Answer:
    """

# 会話のテンプレートを作成
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "message"],
    template=template,
)

# UIの設定
#user_id = st.text_input("IDを入力してエンターを押してください")

#with st.sidebar:
    #user_api_key = st.text_input(
        #label="OpenAI API key",
        #placeholder="Paste your OpenAI API key",
        #type="password"
    #)
    #os.environ['OPENAI_API_KEY'] = user_api_key
select_model = "gpt-4o"
select_temperature = 0.0

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
        private_key = st.secrets["private_key"].replace('\\n', '\n')
        # Firebase認証情報を設定
        cred = credentials.Certificate({
                "type": "service_account",
                "project_id": "main-r",
                "private_key_id": "ffc5a139288747153f225cc9cf2381c334cda3c3",
                "private_key": private_key,
                "client_email": "firebase-adminsdk-m1jdu@main-r.iam.gserviceaccount.com",
                "client_id": "114270809957158975243",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-m1jdu%40main-r.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
                })
        default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    db_path = f"./vector_student/{user_id}"
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
                response = chain({"message": user_message})
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
                group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_eEVBQ7a0d8iVvq6"
                group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
                st.markdown(f'これで今回の会話は終了です。こちらをクリックしてアンケートに回答してください。: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
            else:
                user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
                st.button("送信", on_click=on_input_change)
        # ターン数に応じた機能を追加
        #--------------------------------------------
        #if "messages" not in st.session_state:
            #st.session_state.messages = []

        #for message in st.session_state.messages:
        # with st.chat_message(message["role"]):
                #st.markdown(message["content"])

        #prompt_input = st.chat_input("入力してください", key="propmt_input")

        #if prompt_input:
            #st.session_state.messages.append({"role": "user", "content": prompt_input})
            #with st.chat_message("user"):
                #st.markdown(prompt_input)

            #with st.chat_message("assistant"):
                #with st.spinner("Thinking..."):
                    #response = chain({"message": prompt_input})
                    #st.markdown(response["answer"])
            
            #st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    else:
        st.error(f"No vector database found for student ID {user_id}.")


