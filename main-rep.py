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
    さらに、この会話では「エピソード記憶」を積極的に話題に出して会話してほしいです。エピソード記憶という言葉の意味は以下に示します。
    # エピソード記憶とは、人間の記憶の中でも特に個人的な経験や出来事を覚える記憶の種類の一つです。エピソード記憶は、特定の時間と場所に関連する出来事を含む記憶であり、過去の個人的な経験を詳細に思い出すことができる記憶を指します。
    エピソード記憶を参照して話題に出すというのは、例えば、「あの日に遊園地に行ったときに乗ったあのアトラクションで感じたあの感情は今の感情に似ているね」といった具合です。
    敬語は使わないでください。私の友達になったつもりで砕けた口調で話してください。
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
            "project_id": "main-rep",
            "private_key_id": "e147fc6eb46802a0c41d123b742c3b73060044cd",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCSOOhIDg/CwYqZ\na/kWSlo8NrS5o3yYAzcevB1/ePSRGEF2kPW/yhtLECWdPKEIdbCP+ZV4GfC9ILWp\nEC9M3DKEaoCYzB0w6C23XB/kp30JxZ2NADePu8HLAwO+Ze+sA/voOItsS0VYZENv\ndrcS1eM13jwGMDb08RKOyV2Gb8j83XP0lSM5Z/3Tj4eOu8J6zmRUbLBxkS0shaHf\nbmBVJ2a1f/sd3pfsCoOL8EYGeDJV6NpSA2NgA5QjInS0WAeD6ZtAmvAyrxnysjqW\nl/69+XeM4/XT/paEMztI8/AxqCvfizY30/LPkjWf8G9qCt9GzQTIcvxGPjuROt44\n1FhArjuPAgMBAAECggEAAoW1yAxqDC9/fw8EQHSu4swEyRD/ZLXlfQO8tIa3HUug\nUXXFr3XCi8RvCavSkU1KaulgwS4dX33RdGWhkz9XJb6akkgvFrlilH+j0zNqbkE4\n6wY+bx3LhX97UIz8meiFr2dOTGNuAkzUdvdCyDHlVfgL7AVn5UjFS65AjUGfY3E9\n0aUGiZ0i8rc1RheNpOdydhOzpapy8gN+JXuB4/CbgOXbQIQF23IbF/JYizie90bX\nmJhvBN8jDptz9LdG+6+rbsFG7qkEBiv4VLQA+9nsDHPFClgXjK8bCA8e+yocrqBE\nOowcarX1YgQIoiypcfl03SPH0xDtpP6aglu0xzx4eQKBgQDH0lZGBzg7HKyIhpsT\nF39DahlVHMFBqlvaNgpTnSYEUnY4HzTmdXPYeAIWdhS+lW3SVo1ZqR87TwB4rXQp\nxht33Vtvd5OHwjkZVw9POzsPrYxlWOAjd/Ei4ZOVrqCqiohh3kAijw/DivC5PDlT\nH2YPhGJcZLyvCh43RPNLCzEZlwKBgQC7VORbS9lbQsvmnt5IQBNgbb32l3rACe7d\n+PPfjeLwmGYBaQgJrAnr8YYihy4Ux6g8iE4oN0DBDYkBFXF28SynT5ze1Z0rysyT\nVshGeLApg6vrCNiAO40AP/CXlpiUhrpmHJX5DZY9GVH96n8X13z1z2lJhoedh2U6\nYRd7RqR8yQKBgCH0G7TKhUOGG8sXFEKqO3W5EZTkzfHagMETba3Hhi411OC0bMi8\nLHMb8T7f0aaQvZiAHISfuC7bvbjDwHlYsFItytul9eublRg5MuDDr8V6N+EAuRVN\nzCuhKPLGOYbBA2ud2EgHByay9TSEhzkYnL8GP7BzbZxQm9HIZY7a1/0BAoGAfq3T\nlqFeJMaw2A2Kx1T9RXIwybZ5/a855sVZNU3fr09/e2ipVNEQDIvRZzv+v3KcpAKQ\nx8VFdsdOZHs5tXM1/RZrQI03sct8OA6xdGZcylYORew/a8fZe9fBPOFL4PSSzEZ8\nbGTxufOLbKfMtjS0fg16Z4wf3TkYDThnBqgox3ECgYEAroDXXjb87XYAo4nlexTs\njawXhCVB0dgp+/FJIB4XUyGoao+FRcHb0i7z9K6yQW5LutEQZoe6rmyL37u1f7m7\nbmBskuFdHFyxj2M6ezRpLD2orME8/JzDWM+HhkMm1BNulKZh1qiSmpVl6Z/OzPxE\n4rPAI/+FCOnO0y41rhivaWs=\n-----END PRIVATE KEY-----\n",
            "client_email": "firebase-adminsdk-yhfcb@main-rep.iam.gserviceaccount.com",
            "client_id": "108027337415916591787",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-yhfcb%40main-rep.iam.gserviceaccount.com",
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


