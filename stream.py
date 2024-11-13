import streamlit as st
import time

# 表示するテキスト
text = "Hello, Streamlit! This is a typewriter effect."

# ジェネレーター関数を定義
def typewriter(text):
    for char in text:
        yield char
        time.sleep(0.1)  # 各文字の表示間隔を調整

# ボタンを配置し、クリック時にストリームを開始
if st.button("タイプライターエフェクトを開始"):
    st.write_stream(typewriter(text))
