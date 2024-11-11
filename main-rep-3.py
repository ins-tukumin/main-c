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
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import pytz

# 現在時刻 now
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

if "initialized" not in st.session_state:
    st.session_state['initialized'] = False
    initial_message = "今日の振り返りをしよう！今日はどんな一日だった？"
    st.session_state.initge.append(initial_message)
    st.session_state['initialized'] = True

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
                position: fixed;
                }
                footer {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if user_id:
    if not firebase_admin._apps:
        # Firebase認証情報を設定
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
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    db_path = f"./vector/{user_id}"
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        database = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
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

        # メインプロンプトテンプレートを設定
        custom_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "context_date", "question"],
            template="""
                今日の出来事を振り返って、ユーザーに自由に感想を語ってもらいましょう。適度な問いかけを行って、会話を促進してください。
                私の日記情報（{context_date} に記入されたもの）も添付します。
                この日記を読んで、私の事をよく理解した上で会話してください。
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
        )

        # 各ドキュメントに個別にメタデータ（context_date）を含むためのテンプレートを設定
        document_combine_prompt = PromptTemplate(
            input_variables=["context", "context_date"],
            template="日記情報（{context_date} に記入）: {context}"
        )

        # ConversationalRetrievalChain の設定
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": custom_prompt,
                "document_prompt": document_combine_prompt
            }
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
            chat_history = memory.load_memory_variables({})["chat_history"]

            with st.spinner("相手からの返信を待っています。。。"):
                # chain 呼び出し時にソースドキュメントを取得し、メタデータと応答を取得
                #context_temp = ""
                #context_date_temp = ""
                result = chain({
                    "question": user_message, 
                    "chat_history": chat_history
                })

                # ソースドキュメントから context と context_date を取得
                if 'source_documents' in result:
                    source_docs = result['source_documents']
                    if source_docs:
                        # context はドキュメントの本文
                        context = source_docs[0].page_content
                        # context_date はメタデータから取得、なければデフォルト値を設定
                        context_date = source_docs[0].metadata.get("date", "日付不明")
                    else:
                        context = ""
                        context_date = "日付不明"
                else:
                    context = ""
                    context_date = "日付不明"

                # context と context_date を chain に渡して応答を生成
                response = chain({
                    "question": user_message,
                    "chat_history": chat_history,
                    "context": context,
                    "context_date": context_date  # 必ず指定
                })

                response_text = response["answer"]

            # 会話履歴を更新
            st.session_state.past.append(user_message)
            st.session_state.generated.append(response_text)
            st.session_state.user_message = ""

            # Firebase Firestoreに会話を保存
            Agent_1_Human_Agent = "Human" 
            Agent_2_AI_Agent = "AI" 
            doc_ref = db.collection(str(user_id)).document(str(now))
            doc_ref.set({
                Agent_1_Human_Agent: user_message,
                Agent_2_AI_Agent: response_text
            })

        # 会話履歴の表示
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            message(st.session_state.initge[0], key="init_greeting_plus", avatar_style="micah")
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=str(i), avatar_style="adventurer", seed="Nala")
                key_generated = str(i) + "keyg"
                message(st.session_state.generated[i], key=str(key_generated), avatar_style="micah")

        # 入力エリアと送信ボタン
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
