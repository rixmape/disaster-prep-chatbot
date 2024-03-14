# fmt: off

import json
import os
import time
from operator import itemgetter

import firebase_admin
import streamlit as st
import yaml
from firebase_admin import credentials, firestore, storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.get("LANGCHAIN_ENDPOINT", "https://api.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "default")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

CONFIG_FILE_PATH = "config.yaml"
DOCUMENTS_DIRECTORY = "documents"

# fmt: on


def load_configuration_file():
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_chatbot():
    st.session_state.config = load_configuration_file()
    st.write(st.session_state.config["descriptions"]["app"])
    st.subheader("How should I answer your questions?")

    column1, column2 = st.columns([0.4, 0.6])
    persona_choice = None

    with column2:
        persona_choice = st.selectbox(
            label="Select a persona:",
            options=[
                persona["traits"]
                for persona in st.session_state.config["personas"]
            ]
            + ["Custom"],
            label_visibility="collapsed",
            format_func=lambda x: x.title(),
        )

        with st.container(border=True):
            st.markdown("**Instruction:**")
            if persona_choice == "Custom":
                persona_description = st.text_area(
                    label="Chatbot Persona Description:",
                    placeholder="Describe the chatbot's persona...",
                    height=100,
                    label_visibility="collapsed",
                )
            else:
                selected_persona = next(
                    persona
                    for persona in st.session_state.config["personas"]
                    if persona["traits"] == persona_choice
                )
                persona_description = selected_persona["instruction"].strip()
                st.markdown(persona_description)
            st.session_state.persona_desc = persona_description

    with column1, st.container(border=True):
        st.image(
            (
                "images/default.png"
                if persona_choice == "Custom"
                else selected_persona["image"]
            ),
            use_column_width=True,
        )

    if st.button("Start chatting!"):
        with st.status("Initializing chatbot...", expanded=True):
            initialize_chatbot()
        st.session_state.is_configured_by_user = True
        st.rerun()


def initialize_document_retriever(document_filenames):
    documents = []

    for filename in document_filenames:
        document_loader = TextLoader(filename)
        documents.extend(document_loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    document_splits = text_splitter.split_documents(documents)
    document_embeddings = OpenAIEmbeddings()
    vector_database = Chroma.from_documents(
        document_splits,
        document_embeddings,
    )

    return vector_database.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 4},
    )


def contextualize_prompt(input):
    if not input["history"]:
        return input["question"]

    # FIX: Can't access session state. Hardcoding the prompt for now.
    prompt = (
        "Given a chat history and the latest user question which might"
        " reference context in the chat history, formulate a standalone"
        " question which can be understood without the chat history. Do"
        " not answer the question, just reformulate only if needed."
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return prompt_template | ChatOpenAI() | StrOutputParser()


def format_documents_for_prompt(documents):
    context = "\n\n".join(
        [
            f'Document {i}:\n\n"""\n{doc.page_content}\n"""'
            for i, doc in enumerate(documents, start=1)
        ]
    )
    return f"Use the following documents to answer the query.\n\n{context}"


def initialize_chatbot():
    st.write("💬 Initializing message history...")
    st.session_state.message_history = StreamlitChatMessageHistory(
        key="messages",
    )
    if not st.session_state.messages:
        initial_message = st.session_state.config["prompts"]["initial"].strip()
        st.session_state.message_history.add_ai_message(initial_message)

    st.write("🌐 Initialize cloud connection...")
    if not firebase_admin._apps:
        firebase_auth = dict(st.secrets["FIREBASE_AUTH"])
        firebase_credentials = credentials.Certificate(firebase_auth)
        firebase_admin.initialize_app(
            firebase_credentials,
            {"storageBucket": "streamlit-chatbot-6ee28.appspot.com"},
        )

    st.write("📢 Connecting to user feedback database...")
    firestore_client = firestore.client()
    st.session_state.feedback_database = firestore_client.collection("feedback")

    st.write("📄 Downloading relevant documents...")
    os.makedirs(DOCUMENTS_DIRECTORY, exist_ok=True)
    storage_bucket = storage.bucket()
    storage_blobs = list(storage_bucket.list_blobs())
    document_filenames = []
    for blob in storage_blobs:
        filename = f"{DOCUMENTS_DIRECTORY}/{blob.name}"
        blob.download_to_filename(filename)
        document_filenames.append(filename)

    st.write("🔍 Setting up document retriever...")
    document_retriever = initialize_document_retriever(document_filenames)

    st.write("🔗 Setting up chatbot pipeline...")
    chatbot_instruction = " ".join(
        [
            st.session_state.config["prompts"]["main_instruction"].strip(),
            st.session_state.persona_desc,
            "{context}",
        ]
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", chatbot_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    rag_chain_from_documents = (
        RunnablePassthrough.assign(
            context=(lambda docs: format_documents_for_prompt(docs["context"]))
        )
        | prompt_template
        | ChatOpenAI()
        | StrOutputParser()
    )
    st.session_state.chatbot = RunnableParallel(
        {
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "context": contextualize_prompt | document_retriever,
        }
    ).assign(answer=rag_chain_from_documents)

    st.write("✨ Finishing chatbot configuration...")


def convert_chat_history_to_json():
    return json.dumps(
        [
            {"type": message.type, "content": message.content}
            for message in st.session_state.messages
        ],
        indent=4,
    )


def configure_sidebar():
    with st.sidebar:
        display_helpful_info()
        st.divider()
        display_feedback_form()


def display_helpful_info():
    st.title("💡 Helpful Information")
    st.write(st.session_state.config["descriptions"]["app"])

    with st.expander("Predefined commands"):
        commands = st.session_state.config["commands"]
        for command_name, command_info in commands.items():
            st.markdown(
                f":green[**{command_name}**] : {command_info['description']}\n\n"
                "Sample usage:\n\n"
                f"\t/{command_name} {command_info['arg']}\n\n"
            )

    chat_history_json = convert_chat_history_to_json()
    filename = f"conversation_{int(time.time())}.json"
    st.download_button(
        label="Download Conversation as JSON",
        data=chat_history_json,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
        type="primary",
    )


def display_feedback_form():
    st.title("📢 Feedback")
    st.write(st.session_state.config["descriptions"]["feedback"])
    feedback = dict(subject="", content="", history="")

    feedback["subject"] = st.selectbox(
        "Subject",
        options=[
            "💭 General feedback",
            "🌟 Feature request",
            "🚨 Bug report",
            "📢 Other",
        ],
    )
    feedback["content"] = st.text_area("User Feedback", height=100)
    include_chat_history = st.checkbox("Include chat history in feedback")
    if include_chat_history:
        feedback["history"] = convert_chat_history_to_json()

    if st.button("Submit", type="primary"):
        if feedback:
            feedback["timestamp"] = firestore.SERVER_TIMESTAMP
            st.session_state.feedback_database.add(feedback)
            st.success("Feedback submitted successfully!", icon="🚀")
        else:
            st.error("Give feedback before submitting.", icon="🙀")


def interpret_slash_command(prompt):
    valid_commands = st.session_state.config["commands"]
    if prompt.startswith("/"):
        command, argument = prompt.split(" ", 1)
        command = command.replace("/", "")
        if not command in valid_commands.keys():
            return prompt
        interpreted_prompt = (
            f"{valid_commands[command]['description']}\n\n" f"Query: {argument}"
        )
        return interpreted_prompt
    return prompt


def configure_chat():
    for message in st.session_state.messages:
        st.chat_message(message.type).write(message.content)

    if user_input := st.chat_input():
        interpreted_input = interpret_slash_command(user_input)
        st.chat_message("human").write(interpreted_input)
        response = st.session_state.chatbot.invoke(
            {
                "question": interpreted_input,
                "history": st.session_state.messages,
            }
        )

        ai_message = st.chat_message("ai")
        ai_message.write(response.get("answer"))

        file_citations = response.get("context")
        citations_container = ai_message.expander(
            f"File Citations ({len(file_citations)}):",
            expanded=False,
        )
        for citation in file_citations:
            source = citation.metadata.get("source")
            content = citation.page_content.replace("#", "")
            content = "\n".join([f"> {line}" for line in content.split("\n")])
            citations_container.markdown(f"**{source}**\n{content}")

        st.session_state.message_history.add_user_message(interpreted_input)
        st.session_state.message_history.add_ai_message(response.get("answer"))


if __name__ == "__main__":
    st.set_page_config(page_title="Disaster Preparedness Bot", page_icon="🐱‍🚀")
    st.title("🐱‍🚀 Disaster Preparedness Bot")

    st.session_state.setdefault("is_configured_by_user", False)

    if not st.session_state.is_configured_by_user:
        configure_chatbot()
    else:
        configure_sidebar()
        configure_chat()
