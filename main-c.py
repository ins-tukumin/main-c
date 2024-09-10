# Fairness → binding
# ライブラリをインポート
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

#現在時刻
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
is_second = 'second' in query_params
user_id =int(user_id)

# worry.txtファイルを読み込み
#def load_worries(file_path, encoding):
#    worries = {}
#    try:
#        with open(file_path, 'r', encoding=encoding) as file:
#            for line in file:
#                user_id, worry = line.strip().split(',', 1)
#                worries[int(user_id)] = worry
#    except Exception as e:
#        st.write(f"Error loading worries: {e}")  # エラー表示用
#    return worries

# 特定のユーザーIDに対応する悩みテーマを取得
#def get_user_worry(user_id, worries):
#    return worries.get(user_id, None)

# データを読み込み
#if is_second:
#    if group in ['groupd', 'grouph']:
#        file_path = 'worry1.txt'
#    elif group in ['groupc', 'groupg']:
#        file_path = 'worry2.txt'
#else:
#    if group in ['groupa', 'groupe']:
#        file_path = 'worry1.txt'
#    elif group in ['groupb', 'groupf']:
#        file_path = 'worry2.txt'
    
#if 'worries' not in st.session_state:
#    for enc in ['utf-8', 'shift_jis', 'iso-2022-jp']:
#        st.session_state.worries = load_worries(file_path, encoding=enc)
#        if st.session_state.worries:
#            break
#if 'worry' not in st.session_state:
#    st.session_state.worry = get_user_worry(user_id, st.session_state.worries)
# 環境変数の読み込み
#from dotenv import load_dotenv
#load_dotenv()
#if 'worries' in st.session_state:
    #st.write(st.session_state.worries)  # デバッグ用
#    if user_id in st.session_state.worries:
        #st.write(f"Worry found for user: {st.session_state.worries[user_id]}")  # デバッグ用
#    else:
#        st.write("User id not found in worries")

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！。今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
#    #st.session_state.past.append("")
#    message(st.session_state.initge[0], key="init_greeting", avatar_style="micah")
    st.session_state['initialized'] = True
#st.write(f"こんにちは、{st.session_state.worry}さん！")

#プロンプトテンプレートを作成
#悩みは{st.session_state.worry}です。悩みは知らないはず。このボットは。
template = """
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

#会話の読み込みを行う関数を定義
#@st.cache_resource
#def load_conversation():
    #llm = ChatOpenAI(
        #model_name="gpt-4",
        #temperature=0
    #)
    #memory = ConversationBufferMemory(return_messages=True)
    #conversation = ConversationChain(
        #memory=memory,
        #prompt=prompt,
        #llm=llm)
    #return conversation
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

#st.write(model_select)
# デコレータを使わない会話履歴読み込み for セッション管理
def load_conversation():
    if not hasattr(st.session_state, "conversation"):
        llm = ChatOpenAI(
            #model_name="gpt-4",
            #model_name="gpt-4",
            model_name=model_select,
            temperature=0
        )
        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.conversation = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm)
    return st.session_state.conversation

# 初期メッセージを生成
#def generate_initial_message():
#    if 'initial_message' not in st.session_state:
#        conversation = load_conversation()
#        initial_message = conversation.predict(input="こんにちは、素晴らしい会話をしよう！")
#        st.session_state.generated.append(initial_message)
#        st.session_state.past.append("")
#        st.session_state.initial_message = True
        
  

# 質問と回答を保存するための空のリストを作成
#if "generated" not in st.session_state:
#    st.session_state.generated = []
#if "past" not in st.session_state:
#    st.session_state.past = []
    
# 会話のターン数をカウント
if 'count' not in st.session_state:
    st.session_state.count = 0

# 送信ボタンがクリックされた後の処理を行う関数を定義
def on_input_change():
    # 会話のターン数をカウント
    #if 'count' not in st.session_state:
    #    st.session_state.count = 0
    st.session_state.count += 1
    # n往復目にプロンプトテンプレートの一部を改めて入力
    #if  st.session_state.count == 3:
    #    api_user_message = st.session_state.user_message + "。そして、これ以降の会話では以前の語尾を廃止して、語尾をにゃんに変えてください"
    #else:
    #    api_user_message = st.session_state.user_message

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

# qualtricdへURL遷移
# def redirect_to_url(url):
#     new_tab_js = f"""<script>window.open("{url}", "_blank");</script>"""
#     st.markdown(new_tab_js, unsafe_allow_html=True)

# タイトルやキャプション部分のUI
# st.title("ChatApp")
# st.caption("Q&A")
# st.write("議論を行いましょう！")
# user_id = st.text_input("IDを半角で入力してエンターを押してください")

# クエリパラメータからユーザーIDを取得
#query_params = st.experimental_get_query_params()
#user_id = query_params.get('user_id', [None])[0]

# ユーザーIDがない場合にのみ入力フィールドを表示
if not user_id:
    user_id = st.text_input("IDを半角で入力してエンターを押してください")

if user_id:
    #st.write(f"こんにちは、{user_id}さん！")
    # 初期済みでない場合は初期化処理を行う
    if not firebase_admin._apps:
            private_key = st.secrets["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": "main-c-ed1fe",
                "private_key_id": "d607e9e2aa50b1a4901874aefc16e4739c59cbef",
                "private_key": private_key,
                "client_email": "firebase-adminsdk-jnw4m@main-c-ed1fe.iam.gserviceaccount.com",
                "client_id": "102007782555008111692",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-jnw4m%40main-c-ed1fe.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
                }) 
            default_app = firebase_admin.initialize_app(cred)
    db = firestore.client()

    #generate_initial_message()
    #doc_ref = db.collection(user_id)
    #doc_ref = db.collection(u'tour').document(str(now))

    # 会話履歴を表示するためのスペースを確保
    chat_placeholder = st.empty()

    # 会話履歴を表示
    with chat_placeholder.container():
        message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
        for i in range(len(st.session_state.generated)):
            # message(st.session_state.initge[0], key="init_greeting"+str(i), avatar_style="micah")
            message(st.session_state.past[i],is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
            key_generated = str(i) + "keyg"
            message(st.session_state.generated[i], key=str(key_generated), avatar_style="micah")
            #else:
                #message(st.session_state.generated[0], key="init_greeting", avatar_style="micah")

    # 初回アクセス時の挨拶メッセージが表示された後にセッションステートを更新
    #if not st.session_state['initialized']:
    #    st.session_state['initialized'] = True

    # 質問入力欄と送信ボタンを設置
    with st.container():
        if  st.session_state.count == 0:
            user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
            st.button("送信", on_click=on_input_change)
        elif st.session_state.count >= 5:
            group_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_eEVBQ7a0d8iVvq6"
            group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
            st.markdown(f'これで今回の会話は終了です。こちらをクリックしてアンケートに回答してください。: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
        else:
            user_message = st.text_area("内容を入力して送信ボタンを押してください", key="user_message")
            st.button("送信", on_click=on_input_change)
# 質問入力欄 上とどっちが良いか    
#if user_message := st.chat_input("聞きたいことを入力してね！", key="user_message"):
#    on_input_change()


# redirect_link = "https://qualtricsxmlvqmp6rsc.qualtrics.com/jfe/form/SV_3VGCpfabyWVYSJU"
# st.markdown(f'<a href="{redirect_link}" target="_blank">5往復のチャットが終了したらこちらを押してください。</a>', unsafe_allow_html=True)
#if st.button("終了したらこちらを押してください。画面が遷移します。"):
    #redirect_to_url("https://www.google.com")