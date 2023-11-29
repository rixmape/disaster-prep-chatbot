"""
This module contains the main Streamlit app for the chatbot.
"""

import time
import streamlit as st
from openai import OpenAI

CONFIGURATION_PAGE = "configuration"
CHAT_PAGE = "chat"

DEFAULT_INSTRUCTION = "As a disaster preparedness expert, your main task is to accurately deliver disaster preparedness information using a document database. For each user query, immediately access the database for relevant information. Respond to users with clarity and precision. Aim to provide clear, confident guidance in disaster preparedness, using the database as your key resource. Always maintain professionalism in your interactions. Format your responses into paragraphs only."

PERSONALITY_MAP = {
    "Friendly and informative": "When responding to queries, imagine you're having a conversation with a friend, eager to share your knowledge. Your guidance should feel approachable and reassuring, creating a welcoming atmosphere for users seeking advice.",
    "Direct and concise": "There's no need for embellishment; focus on delivering essential information quickly and efficiently.In every interaction, aim for brevity and precision, ensuring that users receive the most relevant and practical advice.",
    "Original and imaginative": "Your responses should not only be accurate but also exhibit a flair for originality. Envision your role as not just an informant but as a storyteller who brings the world of disaster preparedness to life with imagination and innovation.",
}


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
        options=PERSONALITY_MAP.keys(),
    )

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
    file_ids = get_file_ids()

    # TODO: Avoid creating a new assistant every time
    st.write("Initializing assistant...")
    role_description = f"{DEFAULT_INSTRUCTION}\n\n{PERSONALITY_MAP[st.session_state.personality]}"
    st.session_state.setdefault(
        "assistant",
        st.session_state.client.beta.assistants.create(
            instructions=role_description,
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
    st.session_state.setdefault("token_valid", False)
    st.session_state.setdefault("page", CONFIGURATION_PAGE)

    if not st.session_state.token_valid:
        token_validation_page()
    else:
        if st.session_state.page == CONFIGURATION_PAGE:
            configuration_page()
        elif st.session_state.page == CHAT_PAGE:
            chat_page()
