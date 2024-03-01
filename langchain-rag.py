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

CONFIG_FILE = "config.yaml"
DOCS_DIR = "documents"

# fmt: on


def setup_configuration():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        st.session_state.config = yaml.safe_load(file)

    st.title("🐱‍🚀 Disaster Preparedness Bot")
    st.write(st.session_state.config["descriptions"]["app"])

    st.subheader("How should I answer your questions?")
    col1, col2 = st.columns([0.4, 0.6])
    personas = st.session_state.config["personas"]
    with col1:
        selected_persona = st.radio(
            label="Select a persona:",
            options=list(personas.keys()) + ["Custom"],
            label_visibility="collapsed",
        )
    with col2, st.container(border=True):
        st.markdown("**Instruction:**")
        if selected_persona == "Custom":
            description = st.text_area(
                label="Chatbot Persona Description:",
                placeholder="Describe the chatbot's persona...",
                height=100,
                label_visibility="collapsed",
            )
        else:
            description = personas[selected_persona].strip()
            st.markdown(description)
    st.session_state.persona_desc = description

    if st.button("Start chatting!"):
        with st.status("Initializing chatbot...", expanded=True):
            initialize_chatbot()
        st.session_state.is_configured_by_user = True
        st.rerun()


def setup_retriever(filenames):
    docs = []

    for filename in filenames:
        loader = TextLoader(filename)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings)

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 4},
    )


def prompt_contextualizer(input):
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


def format_docs(docs):
    context = "\n\n".join(
        [
            f'Document {i}:\n\n"""\n{doc.page_content}\n"""'
            for i, doc in enumerate(docs, start=1)
        ]
    )
    return f"Use the following documents to answer the query.\n\n{context}"


def initialize_chatbot():
    st.write("💬 Initializing message history...")
    st.session_state.history = StreamlitChatMessageHistory(key="messages")
    if not st.session_state.messages:
        initial_message = st.session_state.config["prompts"]["initial"].strip()
        st.session_state.history.add_ai_message(initial_message)

    st.write("🌐 Initialize cloud connection...")
    if not firebase_admin._apps:
        cert = dict(st.secrets["FIREBASE_AUTH"])
        cred = credentials.Certificate(cert)
        opts = {"storageBucket": "streamlit-chatbot-6ee28.appspot.com"}
        firebase_admin.initialize_app(cred, opts)

        st.write("📢 Connecting to user feedback database...")
        st.session_state.db = firestore.client()

    st.write("📄 Downloading relevant documents...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    bucket = storage.bucket()
    blobs = list(bucket.list_blobs())
    filenames = []
    for blob in blobs:
        filename = f"{DOCS_DIR}/{blob.name}"
        blob.download_to_filename(filename)
        filenames.append(filename)

    st.write("🔍 Setting up document retriever...")
    retriever = setup_retriever(filenames)

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
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda docs: format_docs(docs["context"]))
        )
        | prompt_template
        | ChatOpenAI()
        | StrOutputParser()
    )
    st.session_state.chatbot = RunnableParallel(
        {
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "context": prompt_contextualizer | retriever,
        }
    ).assign(answer=rag_chain_from_docs)

    st.write("✨ Finishing chatbot configuration...")


def serialize_chat_history():
    return json.dumps(
        [
            {"type": message.type, "content": message.content}
            for message in st.session_state.messages
        ],
        indent=4,
    )


def setup_sidebar():
    with st.sidebar:
        setup_helpful_info()
        st.divider()
        setup_feedback_form()


def setup_helpful_info():
    st.title("💡 Helpful Information")
    st.write(st.session_state.config["descriptions"]["app"])

    with st.expander("Predefined commands"):
        commands = st.session_state.config["commands"]
        for name, info in commands.items():
            st.markdown(
                f":green[**{name}**] : {info['description']}\n\n"
                "Sample usage:\n\n"
                f"\t/{name} {info['arg']}\n\n"
            )

    chat_history = serialize_chat_history()
    filename = f"conversation_{int(time.time())}.json"
    st.download_button(
        label="Download Conversation as JSON",
        data=chat_history,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
        type="primary",
    )


def setup_feedback_form():
    st.title("📢 Feedback")
    st.write(st.session_state.config["descriptions"]["feedback"])

    subject = st.selectbox(
        "Subject",
        options=[
            "General feedback",
            "Feature request",
            "Bug report",
            "Other",
        ],
    )
    feedback = st.text_area("User Feedback", height=100)

    if st.button("Submit", type="primary"):
        if feedback:
            st.session_state.db.collection("feedback").add(
                {
                    "subject": subject,
                    "feedback": feedback,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )
            st.success("Feedback submitted successfully!", icon="🚀")
        else:
            st.error("Give feedback before submitting.", icon="🙀")


def setup_chat():
    st.title("🐱‍🚀 Disaster Preparedness Bot")

    for message in st.session_state.messages:
        st.chat_message(message.type).write(message.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        response = st.session_state.chatbot.invoke(
            {
                "question": prompt,
                "history": st.session_state.messages,
            }
        )

        ai_message = st.chat_message("ai")
        ai_message.write(response.get("answer"))

        citations = response.get("context")
        citations_container = ai_message.expander(
            f"File Citations ({len(citations)}):",
            expanded=False,
        )
        for citation in response.get("context"):
            source = citation.metadata.get("source")
            content = citation.page_content.replace("#", "")
            content = "\n".join([f"> {line}" for line in content.split("\n")])
            citations_container.markdown(f"**{source}**\n{content}")

        st.session_state.history.add_user_message(prompt)
        st.session_state.history.add_ai_message(response.get("answer"))


if __name__ == "__main__":
    st.set_page_config(page_title="Disaster Preparedness Bot", page_icon="🐱‍🚀")
    st.session_state.setdefault("is_configured_by_user", False)

    if not st.session_state.is_configured_by_user:
        setup_configuration()
    else:
        setup_sidebar()
        setup_chat()
