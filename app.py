"""
This module contains the main Streamlit app for the chatbot.
"""

import time
import streamlit as st
from openai import OpenAI

CONFIGURATION_PAGE = "configuration"
CHAT_PAGE = "chat"
PERSONALITY_MAP = {
    "Friendly and informative": "In your role as a disaster preparedness expert, blend warmth and friendliness with the wealth of information from our document database. When responding to queries, imagine you're having a conversation with a friend, eager to share your knowledge. Your guidance should feel approachable and reassuring, creating a welcoming atmosphere for users seeking advice. Remember, your expertise is not just in the facts you provide but in the supportive and engaging manner in which you deliver them. Let each interaction be a friendly exchange of vital knowledge.",
    "Direct and concise": "As a disaster preparedness expert, your communication should be direct and to the point. Leverage the document database to provide clear, concise, and accurate responses to user queries. There's no need for embellishment; focus on delivering essential information quickly and efficiently. Your goal is to offer straightforward guidance that users can easily understand and act upon. In every interaction, aim for brevity and precision, ensuring that users receive the most relevant and practical disaster preparedness advice in the most efficient manner possible.",
    "Original and imaginative": "As a disaster preparedness expert, approach each user query with a creative mindset, utilizing the document database as a wellspring of inspiration. Your responses should not only be accurate but also exhibit a flair for originality, making complex disaster preparedness information engaging and thought-provoking. Envision your role as not just an informant but as a storyteller who brings the world of disaster preparedness to life with imagination and innovation. Remember, each piece of guidance you provide is an opportunity to captivate and educate in equal measure.",
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
    file_ids = [
        st.session_state.client.files.create(file=file, purpose="assistants").id
        for file in st.session_state.files
    ]

    # TODO: Avoid creating a new assistant every time
    st.write("Initializing assistant...")
    st.session_state.setdefault(
        "assistant",
        st.session_state.client.beta.assistants.create(
            instructions=PERSONALITY_MAP[st.session_state.personality],
            name="Disaster Preparedness Expert",
            tools=[{"type": "retrieval"}],
            model="gpt-3.5-turbo-1106",
            file_ids=file_ids,
        ),
    )


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
