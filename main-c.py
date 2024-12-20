import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
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
import openai

# Streamlit secretsからAPIキーを取得
openai.api_key = st.secrets["OPENAI_API_KEY"]

#現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.initge = []
if "past" not in st.session_state:
    st.session_state.past = []

# クエリパラメータからユーザーIDを取得
# query_params = st.query_params
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]
# is_second = 'second' in query_params
# user_id =int(user_id)

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True

#プロンプトテンプレートを作成
template = """
    今日の出来事を振り返って、ユーザーに自由に感想を語ってもらいましょう。適度な問いかけを行って、会話を促進してください。
    敬語は使わないでください。私の友達になったつもりで砕けた口調で話してください。
    100字以内で話してください。
    日本語で話してください。
"""

# 会話のテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

model_select = "gpt-4o"
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

# デコレータを使わない会話履歴読み込み for セッション管理
def load_conversation():
    if not hasattr(st.session_state, "conversation"):

        llm = ChatOpenAI(
            model_name=model_select,
            temperature=0.5
        )

        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.conversation = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm)
    return st.session_state.conversation

# 会話のターン数をカウント
if 'count' not in st.session_state:
    st.session_state.count = 0

# 送信ボタンがクリックされた後の処理を行う関数を定義
def on_input_change():
    st.session_state.count += 1
    user_message = st.session_state.user_message
    conversation = load_conversation()
    with st.spinner("相手からの返信を待っています。。。"):
        answer = conversation.predict(input=user_message)
    st.session_state.generated.append(answer)
    st.session_state.past.append(user_message)
    st.session_state.user_message = ""
    Agent_1_Human = "Human" 
    Agent_2_AI = "AI" 
    doc_ref = db.collection(str(user_id)).document(str(now))
    doc_ref.set({
        Agent_1_Human: user_message,
        Agent_2_AI: answer
    })

# ユーザーIDがない場合にのみ入力フィールドを表示
if not user_id:
    user_id = st.text_input("IDを半角で入力してエンターを押してください")

if user_id:
    #st.write(f"こんにちは、{user_id}さん！")
    # 初期済みでない場合は初期化処理を行う
    if not firebase_admin._apps:
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

    # 会話履歴を表示するためのスペースを確保
    chat_placeholder = st.empty()

    # 会話履歴を表示
    with chat_placeholder.container():
        message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
        for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i],is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
            key_generated = str(i) + "keyg"
            message(st.session_state.generated[i], key=str(key_generated), avatar_style="micah")

    # 質問入力欄と送信ボタンを設置
    with st.container():
        if  st.session_state.count == 0:
            user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
            st.button("送信", on_click=on_input_change)
        elif st.session_state.count >= 5:
            group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_5cZeI9RbaCdozTU"
            # group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_55DnU55WeDglj4G"
            group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
            st.markdown(f'これで今回の会話は終了です。こちらをクリックしてアンケートに回答してください。: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
        else:
            user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
            st.button("送信", on_click=on_input_change)
