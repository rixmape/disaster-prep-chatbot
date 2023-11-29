"""
This module contains the main Streamlit app for the chatbot.
"""

import time
import streamlit as st
from openai import OpenAI
import yaml


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

    st.session_state.personality = st.selectbox(
        "Select a personality:",
        options=st.session_state.config["personalities"].keys(),
    )

    st.session_state.files = st.file_uploader(
        "Upload some files:",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "html", "md", "pptx"],
    )

    if st.button("Start chatting!"):
        st.session_state.configured = True


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
    file_ids = get_file_ids()

    # TODO: Avoid creating a new assistant every time
    instructions = st.session_state.config["default_instructions"].format(
        personality=st.session_state.config["personalities"][
            st.session_state.personality
        ],
    )
    st.session_state.setdefault(
        "assistant",
        st.session_state.client.beta.assistants.create(
            instructions=instructions,
            name="Disaster Preparedness Expert",
            tools=[{"type": "retrieval"}],
            model="gpt-3.5-turbo-1106",
            file_ids=file_ids,
        ),
    )


def get_file_ids():
    openai_files = {
        file.filename: file.id
        for file in st.session_state.client.files.list().data
    }

    file_ids = []

    for file in st.session_state.files:
        # TODO: Improve file deduplication by checking file contents
        if file.name in openai_files:
            file_id = openai_files[file.name]
            file_ids.append(file_id)
        else:
            file = st.session_state.client.files.create(
                file=file,
                purpose="assistants",
            )
            file_ids.append(file.id)

    return file_ids


def chat_page():
    st.title("Step 3: Let's Chat!")

    if "client" not in st.session_state:
        with (
            st.chat_message("assistant"),
            st.status("Initializing chatbot..."),
        ):
            initialize_chatbot()
            st.rerun()  # Refresh the page to update the session state

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id,
        order="asc",
    )

    for message in messages:
        text = message.content[0].text
        assistant_message = st.chat_message(message.role)
        assistant_message.markdown(text.value)

        citations = [
            annotation
            for annotation in text.annotations
            if annotation.type == "file_citation"
        ]

        if not citations:
            continue  # Don't display the citations section if there are none

        citation_container = assistant_message.expander(
            f"File Citations ({len(citations)})",
            expanded=False,
        )

        for index, citation in enumerate(citations):
            label = f"**{index+1}. {citation.text}:**"
            quote = "\n".join(
                [
                    f"> {line}"
                    for line in citation.file_citation.quote.split("\n")
                ]
            )
            citation_container.markdown(f"{label}\n{quote}")

    if prompt := st.chat_input("What's on your mind?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        message = st.session_state.client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt,
        )

        run = st.session_state.client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )

        with (
            st.chat_message("assistant"),
            st.status("Waiting for response..."),
        ):
            while run.status in ("queued", "in_progress"):
                run = st.session_state.client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread.id,
                    run_id=run.id,
                )
                time.sleep(0.5)

        st.rerun()  # Refresh the page to display new messages


if __name__ == "__main__":
    if "config" not in st.session_state:
        with open("config.yaml", "r", encoding="utf-8") as file:
            st.session_state.config = yaml.safe_load(file)

    st.session_state.setdefault("token_valid", False)
    st.session_state.setdefault("configured", False)

    if not st.session_state.token_valid:
        token_validation_page()
    else:
        if not st.session_state.configured:
            configuration_page()
        else:
            chat_page()
