"""
This module contains the main Streamlit app for the chatbot.
"""

import time
import streamlit as st
from openai import OpenAI

# Initialize session state variables
st.session_state.setdefault("token_valid", False)
st.session_state.setdefault("files", [])

# Show the token input only if the token has not been validated yet
if not st.session_state.token_valid:
    st.title("Token Validation")
    access_token = st.text_input(
        "Enter your access token to continue:",
        type="password",
        help="Get your access token from the app developer.",
    )

    # Check if the entered token is valid
    if access_token and access_token in st.secrets["access_tokens"]:
        st.success("You have successfully entered a valid token!")
        st.session_state.token_valid = True
        st.rerun()
    elif access_token:
        st.error("Please enter a valid token to access the main page.")
else:
    if "client" not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.client = OpenAI(api_key=api_key)

    client = st.session_state.client  # Alias for convenience

    if "thread" not in st.session_state:
        st.session_state.thread = client.beta.threads.create()

    if "messages" not in st.session_state:
        st.session_state.messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id,
            order="asc",
        )

    if "assistant" not in st.session_state:
        assistant_id = st.secrets["openai_assistant_id"]
        st.session_state.assistant = client.beta.assistants.retrieve(
            assistant_id=assistant_id
        )

    sidebar_message = st.sidebar.empty()

    files = st.sidebar.file_uploader(
        "Upload some files:",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "html", "md", "pptx"],
    )

    if files != st.session_state.files:
        sidebar_message.warning("Unsaved files!")

    *_, col = st.sidebar.columns(4)  # Right align the button
    if col.button("Save"):
        if files != st.session_state.files:
            st.session_state.files = files

            file_ids = []
            for file in st.session_state.files:
                assistant_file = client.files.create(
                    file=file,
                    purpose="assistants",
                )
                file_ids.append(assistant_file.id)

            st.session_state.assistant = client.beta.assistants.update(
                assistant_id=st.session_state.assistant.id,
                file_ids=file_ids,
            )

            # Reset the thread to clear the chat history
            st.session_state.thread = client.beta.threads.create()
            st.session_state.messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread.id,
                order="asc",
            )

            sidebar_message.success("Files saved!")
        else:
            sidebar_message.warning("No changes made.")

    st.title("Let's Chat!")
    expander = st.expander("Session State", expanded=False)

    if prompt := st.chat_input("What's on your mind?"):
        message = client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt,
        )

        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )

        while run.status in ("queued", "in_progress"):
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)

        st.session_state.messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id,
            order="asc",
        )

    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)

    expander.json(st.session_state)
