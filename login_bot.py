import streamlit as st
import random

# 各グループに対応するURLを定義します
group_urls = {
    "groupa": "https://chat-c.streamlit.app/",
    "groupb": "https://chat-r.streamlit.app/",
    "groupc": "https://chat-rep.streamlit.app/"
}

# 特別なURLを定義します
special_url = "https://www.google.com"

# participants.txtファイルからIDとグループを読み込む関数
def load_participants(file_path):
    participants = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    user_id, group = line.split(',')
                    participants[user_id] = group
    except FileNotFoundError:
        pass  # ファイルが見つからない場合でもエラーを表示しない

    return participants

# Streamlitアプリケーションのレイアウト
st.title("ログインページ")
st.subheader("後半フェーズ１日目")

# 参加者のリストを読み込む
participants = load_participants('group_assignment.txt')

# ユーザー入力を受け取る
user_id = st.text_input("クラウドワークスID入力してください")
if st.button("ログイン"):
    if user_id:
        if user_id == "xxxx":
            random_number = random.randint(1000000000, 9999999999)
            user_id = f"xxxx{random_number}"
            group="groupx"
            group_url_with_id = f"{special_url}?user_id={user_id}&group={group}"
            st.success(f"ログイン成功: {user_id}")
            st.markdown(f'こちらのURLをクリックしてください: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
        elif user_id in participants:
            group = participants[user_id]
            if group in group_urls:
                group_url = group_urls[group]
                group_url_with_id = f"{group_url}?user_id={user_id}&group={group}"
                st.success(f"ログイン成功: {user_id}")
                st.markdown(f'こちらのURLをクリックしてください: <a href="{group_url_with_id}" target="_blank">リンク</a>', unsafe_allow_html=True)
            else:
                st.error("対応するグループURLが見つかりません。")
        else:
            st.error("無効なIDです。もう一度お試しください。")
    else:
        st.error("学籍番号を入力してください。")