import streamlit as st
import openai

# OpenAI APIキーの設定
# openai.api_key = "your_openai_api_key_here"  # あなたのAPIキーを入力してください

# Streamlitアプリの設定
st.title("シンプルなチャットボット")
st.write("OpenAI APIを使用した簡単なチャットボットです。以下の入力欄にメッセージを入力してください。")

# チャット履歴を保存するためのセッションステートの設定
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ユーザーからの入力
user_input = st.text_input("あなたのメッセージを入力してください:")

# ユーザーの入力がある場合の処理
if user_input:
    # ユーザーのメッセージを保存
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # OpenAI APIを呼び出して応答を取得
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # 使用するモデルを指定
            prompt="\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]]),
            max_tokens=150
        )
        # AIの応答を取得
        reply = response.choices[0].text.strip()
        # 応答をチャット履歴に追加
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        # 応答を表示
        st.text_area("チャットボットの応答:", value=reply, height=200, max_chars=None)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# チャット履歴を表示（オプション）
if st.checkbox("チャット履歴を表示"):
    for msg in st.session_state["messages"]:
        role = "ユーザー" if msg["role"] == "user" else "アシスタント"
        st.write(f"**{role}**: {msg['content']}")
