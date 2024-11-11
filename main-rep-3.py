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
import firebase_admin
from firebase_admin import credentials, firestore
from langchain.schema import BaseRetriever
from pydantic import Field
import datetime
import pytz
import time

# 現在時刻
global now
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# 初期化
if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.initge = []
if "past" not in st.session_state:
    st.session_state.past = []

# クエリパラメータからユーザーIDを取得
query_params = st.experimental_get_query_params()
user_id = query_params.get('user_id', [None])[0]
group = query_params.get('group', [None])[0]

# 初期化メッセージ設定
if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True

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
    {question}
    Answer:
    """
prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
select_model = "gpt-4o"
select_temperature = 0.5

# Firebase設定
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

        # Chromaベクトルデータベースの設定
        database = Chroma(persist_directory=db_path, embedding_function=embeddings)
        chat = ChatOpenAI(model=select_model, temperature=select_temperature)
        retriever = database.as_retriever()

        # CustomRetrieverをBaseRetrieverから継承し、retrieverフィールドを定義
        class CustomRetriever(BaseRetriever):
            retriever: BaseRetriever = Field()

            def __init__(self, retriever: BaseRetriever, **kwargs):
                super().__init__(**kwargs)
                object.__setattr__(self, 'retriever', retriever)

            def get_relevant_documents(self, query):
                docs = self.retriever.get_relevant_documents(query)
                for doc in docs:
                    date_info = doc.metadata.get("date", "不明な日付")
                    doc.page_content = f"【日付: {date_info}】\n{doc.page_content}"
                return docs

        # カスタムリトリーバーを使ってリトリーバーをラップ
        custom_retriever = CustomRetriever(retriever)

        # メモリの設定
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        memory = st.session_state.memory

        # ConversationalRetrievalChainの設定
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=custom_retriever,  # カスタムリトリーバーを指定
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        # 入力を基に応答を取得する
        def on_input_change():
            st.session_state.count += 1
            user_message = st.session_state.user_message
            with st.spinner("相手からの返信を待っています..."):
                response = chain({"question": user_message})
                response_text = response["answer"]

            # 会話履歴に追加
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""

            # Firestoreに保存
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({"Human": user_message, "AI": response_text})

        # 会話の表示
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
                key_generated = str(i) + "keyg"
                message(st.session_state.generated[i], key=str(key_generated), avatar_style="micah")

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
