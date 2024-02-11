"""
This module contains the main Streamlit app for the chatbot.
"""

import os
import time

import streamlit as st
import yaml
from openai import OpenAI


def get_chat_history_str():
    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id,
        order="asc",
    )
    return "\n\n".join(
        [
            f"{message.role.capitalize()}: {message.content[0].text.value}"
            for message in messages
        ]
    )


def setup_sidebar():
    with st.sidebar:
        st.title("Local Disaster Preparedness Chatbot")
        st.markdown(st.session_state.config["app_description"])

        if "filenames" in st.session_state:
            with st.expander("Uploaded files"):
                st.markdown(
                    "\n".join(
                        f"- **{filename}**"
                        for filename in st.session_state.filenames
                    )
                )

        with st.expander("Predefined commands"):
            commands_description = "\n\n".join(
                f":green[**/{command}**]: {expansion.split('.')[0]} ..."
                for command, expansion in st.session_state.config[
                    "command_map"
                ].items()
            )
            st.markdown(commands_description)

        if "client" in st.session_state:
            st.download_button(
                label="Download Conversation",
                data=get_chat_history_str(),
                use_container_width=True,
                type="primary",
            )


def setup_config_page():
    st.title("Configure your chatbot")
    st.session_state.filenames = os.listdir("documents")
    st.session_state.personality = st.selectbox(
        "Select a personality:",
        options=st.session_state.config["personalities"].keys(),
    )

    if st.button("Start chatting!"):
        if st.session_state.filenames:
            st.session_state.configured = True
        else:
            st.error("No files uploaded.")


def initialize_chatbot():
    st.write("Initializing OpenAI client...")
    st.session_state.setdefault(
        "client",
        OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
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
            model=st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo-1106"),
            file_ids=file_ids,
        ),
    )


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
    instruction = st.session_state.config["command_map"].get(command)
    if instruction:
        return f"{instruction}\n\nQuery: {query}"
    else:
        return prompt


def setup_chat_page():
    st.title("Let's Chat!")

    if "client" not in st.session_state:
        with (
            st.chat_message("assistant"),
            st.status("Initializing chatbot..."),
        ):
            initialize_chatbot()
            st.rerun()  # Refresh the page to update the session state

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
    st.session_state.setdefault("configured", False)
    if "config" not in st.session_state:
        with open("config.yaml", "r", encoding="utf-8") as file:
            st.session_state.config = yaml.safe_load(file)

    setup_sidebar()
    if st.session_state.configured:
        setup_chat_page()
    else:
        setup_config_page()
