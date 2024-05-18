import streamlit as st

from chatbot import ChatbotAgent

if __name__ == "__main__":
    st.set_page_config(page_title="🤖 Dipcy, your DRR companion.", page_icon="🐱‍🚀")
    st.title("🤖 Dipcy, your DRR companion.")

    st.session_state.setdefault("chatbot", None)
    if not st.session_state.chatbot:
        st.session_state.chatbot = ChatbotAgent()

    st.session_state.chatbot.run()
