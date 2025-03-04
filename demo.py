import streamlit as st
from streamlit_chat import message

# Streamlit UI 設定
st.set_page_config(page_title="会話デモ", layout="centered")

# 事前に用意した会話リスト（後で変更可）
conversation_history = [
    ("AI", "今日の振り返りをしよう！今日はどんな一日だった？"),
    ("User", "特に何もなかったけど、ちょっと疲れたかも。"),
    ("AI", "そっか、疲れた時はリラックスするのが大事だね。"),
    ("User", "うん。でもやることも多くて、なかなか休めないんだよね。"),
    ("AI", "それは大変だね。少しでも気分転換できるといいね！"),
]

# UI に会話を表示
# st.title("💬 会話デモ")

for i, (speaker, text) in enumerate(conversation_history):
    if speaker == "User":
        message(text, is_user=True, key=f"user_{i}", avatar_style="adventurer", seed="Nala")
    else:
        message(text, key=f"ai_{i}", avatar_style="micah")

# ユーザー入力欄は削除して、UIデモのみ
# st.write("⚡ これはデモ表示です。ユーザー入力はできません。")

