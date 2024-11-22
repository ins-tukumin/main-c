# 本実験後半３日に合わせるためのダミーログインページなので、直接qualtricsURLへ案内してもよい
import streamlit as st
import random

# 特定のURLを定義します
# special_url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_55DnU55WeDglj4G"
special_url = "https://chat-c.streamlit.app/"

# Streamlitアプリケーションのレイアウト
st.title("ログインページ")
st.subheader("1日目")

# ユーザー入力を受け取る
user_id = st.text_input("クラウドワークスIDを入力してください")
if st.button("ログイン"):
    if user_id:
        if user_id == "xxxx":
            random_number1 = random.randint(1000000000, 9999999999)
            user_id = f"xxxx{random_number1}"
            group = "groupx"
        else:
            group = "groupa"  # デフォルトのグループを設定
        group_url_with_id = f"{special_url}?user_id={user_id}"
        #st.success(f"ログイン成功: {user_id}")
        st.markdown(f'こちらのURLをクリックしてください: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
    else:
        st.error("クラウドワークスIDを入力してください。")
