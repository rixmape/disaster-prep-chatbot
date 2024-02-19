"""
This module contains the main Streamlit app for the chatbot.
"""

import json
import os
import time

import firebase_admin
import streamlit as st
import yaml
from firebase_admin import credentials, firestore
from openai import OpenAI


@st.cache
def get_chat_history_json():
    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id,
        order="asc",
    )
    return json.dumps(
        [
            {
                "role": message.role,
                "value": message.content[0].text.value,
            }
            for message in messages
        ]
    )


def setup_sidebar():
    with st.sidebar:
        st.title("💡 Helpful Information")
        st.markdown(st.session_state.config["app_description"])

        with st.expander("Uploaded files"):
            st.markdown(
                "\n".join(
                    f"- **{filename}**"
                    for filename in st.session_state.filenames
                )
            )

        with st.expander("Predefined commands"):
            commands = st.session_state.config["commands"]
            for name, info in commands.items():
                st.markdown(
                    f":green[**{name}**] : {info['description']}\n\n"
                    "Sample usage:\n\n"
                    f"\t/{name} {info['arg']}\n\n"
                )

        st.download_button(
            label="Download Conversation",
            data=get_chat_history_json(),
            file_name=f"chat-history-{time.time()}.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
        )

        st.divider()

        st.title("📢 Feedback")
        setup_feedback_form()


def setup_feedback_form():
    st.write(st.session_state.config.get("feedback_description"))
    subject = st.selectbox(
        "Subject",
        options=[
            "General feedback",
            "Feature request",
            "Bug report",
            "Other",
        ],
    )
    feedback = st.text_area("Feedback", height=100)
    if st.button("Submit", type="primary"):
        st.session_state.db.collection("feedback").add(
            {
                "subject": subject,
                "feedback": feedback,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
        st.success("Feedback submitted successfully!")


def setup_config_page():

    st.write(st.session_state.config["app_description"])

    st.header("Configuration")
    st.session_state.personality = st.selectbox(
        "Select a personality:",
        options=st.session_state.config["personalities"].keys(),
    )

    if st.button("Start chatting!"):
        with st.status("Initializing chatbot...", expanded=True):
            initialize_chatbot()
        st.session_state.configured = True
        st.rerun()


def initialize_chatbot():
    st.write("🤖 Creating chatbot agent...")
    st.session_state.setdefault(
        "client",
        OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
    )

    st.write("💬 Preparing chat history...")
    st.session_state.setdefault(
        "thread",
        st.session_state.client.beta.threads.create(),
    )

    st.write("📄 Reading knowledge database...")
    st.session_state.filenames = os.listdir("documents")
    file_ids = get_file_ids()

    st.write("✨ Setting chatbot personality...")
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
            model=st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo-1106"),
            file_ids=file_ids,
        ),
    )

    st.write("🌐 Connecting to feedback database...")
    if not firebase_admin._apps:  # Avoid reinitializing the app
        cert = dict(st.secrets["FIREBASE_AUTH"])
        cred = credentials.Certificate(cert)
        firebase_admin.initialize_app(cred)
        st.session_state.db = firestore.client()


def get_file_ids():
    openai_files = {
        file.filename: file.id
        for file in st.session_state.client.files.list().data
    }

    file_ids = []
    for filename in st.session_state.filenames:
        # TODO: Improve file deduplication by checking file contents
        if filename in openai_files:
            file_id = openai_files[filename]
            file_ids.append(file_id)
        else:
            assistant_file = st.session_state.client.files.create(
                file=filename,
                purpose="assistants",
            )
            file_ids.append(assistant_file.id)

    return file_ids


def parse_slash_command(prompt):
    command, query = prompt.lstrip("/").split(" ", 1)
    info = st.session_state.config["commands"].get(command)
    if info:
        return (
            f"{info['description']}"
            " Access the uploaded files to search for relevant information."
            " Deliver the details in a clear and concise language.\n\n"
            f" Input: {query}"
        )
    else:
        return prompt


def setup_chat_page():
    # TODO: Add initial message to the thread. Not currently supported by API.
    inital_message = st.session_state.config["initial_message"]
    with st.chat_message("assistant"):
        st.markdown(inital_message)

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id,
        order="asc",
    )

    for message in messages:
        if message.metadata.get("hidden"):
            continue

        text = message.content[0].text
        assistant_message = st.chat_message(message.role)
        assistant_message.markdown(text.value)

        citations = [
            annotation
            for annotation in text.annotations
            if annotation.type == "file_citation"
        ]

        if not citations:
            continue

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
        if prompt.startswith("/"):
            prompt = parse_slash_command(prompt)

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
    st.set_page_config(
        page_title="Disaster Preparedness Bot",
        page_icon=":robot:",
    )

    with open("config.yaml", "r", encoding="utf-8") as file:
        st.session_state.config = yaml.safe_load(file)

    st.title("Disaster Preparedness Bot")

    if "client" in st.session_state:
        setup_sidebar()
        setup_chat_page()
    else:
        setup_config_page()
