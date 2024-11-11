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

# SQLite3モジュール設定
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Firebase設定
import firebase_admin
from firebase_admin import credentials, firestore

# 現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# Streamlitの状態を初期化
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []
if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.generated.append(initial_message)
    st.session_state['initialized'] = True

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

# クエリパラメータからユーザーIDを取得
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]

# プロンプトテンプレート設定
template = """
    今日の出来事を振り返って、ユーザーに自由に感想を語ってもらいましょう。適度な問いかけを行って、会話を促進してください。
    私の日記情報も添付します。
    この日記を読んで、私の事をよく理解した上で会話してください。
    必要に応じて、私の日記に書かれている情報を参照して、私の事を理解して会話してください。
    ただ、”あなたの日記を読んでみると”といったような、日記を読んだ動作を直接示すような言葉は出力に含めないでください。
    さらに、この会話では私の日記に含まれる「エピソード記憶」を適切に会話に盛り込んで話してほしいです。エピソード記憶という言葉の意味は以下に示します。
    # エピソード記憶とは、人間の記憶の中でも特に個人的な経験や出来事を覚える記憶の種類の一つです。エピソード記憶は、特定の時間と場所に関連する出来事を含む記憶であり、過去の個人的な経験を詳細に思い出すことができる記憶を指します。
    また、今日は１１月１１日です。必要に応じて日記の記入日も考慮して自然な会話を心掛けてください。
    敬語は使わないでください。私の友達になったつもりで砕けた口調で話してください。
    100字以内で話してください。
    日本語で話してください。
    私の入力に基づき、次の文脈（<ctx></ctx>で囲まれた部分）とチャット履歴（<hs></hs>で囲まれた部分）と付加されている日記である文脈情報が書かれた日（<dy></dy>で囲まれた部分）を使用して回答してください。:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    {question}
    Answer:
"""


# PromptTemplateの設定
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"], #"entry_date"
    template=template,
)

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
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # ユーザーIDに基づくベクトルデータベースの読み込み
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

        # ConversationalRetrievalChainの設定
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        # 入力と会話履歴の管理
        def on_input_change():
            user_message = st.session_state.user_message
            with st.spinner("応答を生成中..."):
                response = chain({"question": user_message})
                
                # 応答内容の作成
                context_data = []
                for doc in response["context"]:
                    entry_date = doc.metadata.get("date", "日付不明")
                    page_content = doc.page_content
                    context_data.append(f"日付: {entry_date}\n内容: {page_content}")

                # Streamlit上でentry_dateを表示
                st.write(f"Debug - Entry Date: {entry_date}")

                full_context = "\n\n".join(context_data)
                full_prompt = template.format(
                    chat_history=memory.load_memory_variables().get("chat_history", ""),
                    context=full_context,
                    question=user_message
                    # entry_date=entry_date
                )

                # 応答生成
                response_text = chat(full_prompt)
                st.session_state.past.append(user_message)
                st.session_state.generated.append(response_text)
                st.session_state.user_message = ""

                # Firebase Firestoreに会話履歴を保存
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



