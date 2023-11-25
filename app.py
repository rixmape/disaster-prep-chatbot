"""
This module contains the main Streamlit app for the chatbot.
"""

import time
import streamlit as st
from openai import OpenAI

CONFIGURATION_PAGE = "configuration"
CHAT_PAGE = "chat"


def token_validation_page():
    st.title("Step 1: Validate your token")
    access_token = st.text_input(
        "Enter your access token to continue:",
        type="password",
        help="Get your access token from the app developer.",
    )

    if access_token and access_token in st.secrets["access_tokens"]:
        st.success("You have successfully entered a valid token!")
        st.session_state.token_valid = True
        st.rerun()
    elif access_token:
        st.error("Please enter a valid token to access the main page.")


def configuration_page():
    st.title("Step 2: Configure your chatbot")

    st.session_state.files = st.file_uploader(
        "Upload some files:",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "html", "md", "pptx"],
    )

    if st.button("Start chatting!"):
        st.session_state.page = CHAT_PAGE


def initialize_chatbot():
    st.write("Initializing OpenAI client...")
    st.session_state.setdefault(
        "client",
        OpenAI(api_key=st.secrets["openai_api_key"]),
    )

    st.write("Creating thread...")
    st.session_state.setdefault(
        "thread",
        st.session_state.client.beta.threads.create(),
    )

    st.write("Uploading files...")
    file_ids = [
        st.session_state.client.files.create(file=file, purpose="assistants").id
        for file in st.session_state.files
    ]

    st.write("Initializing assistant...")
    st.session_state.setdefault(
        "assistant",
        st.session_state.client.beta.assistants.update(
            assistant_id=st.secrets["openai_assistant_id"],
            file_ids=file_ids,
        ),
    )


def chat_page():
    st.title("Step 3: Let's Chat!")

    if "client" not in st.session_state:
        with st.status("Initializing chatbot..."):
            initialize_chatbot()
            st.rerun()

    if prompt := st.chat_input("What's on your mind?"):
        message = st.session_state.client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt,
        )

        run = st.session_state.client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )

        while run.status in ("queued", "in_progress"):
            run = st.session_state.client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id,
        order="asc",
    )

    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)


if __name__ == "__main__":
    st.session_state.setdefault("token_valid", False)
    st.session_state.setdefault("page", CONFIGURATION_PAGE)

    if not st.session_state.token_valid:
        token_validation_page()
    else:
        if st.session_state.page == CONFIGURATION_PAGE:
            configuration_page()
        elif st.session_state.page == CHAT_PAGE:
            chat_page()
