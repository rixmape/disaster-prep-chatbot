# fmt: off

import json
import os
import time
from operator import itemgetter

import firebase_admin
import streamlit as st
import yaml
from firebase_admin import credentials, firestore
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


def setup_retriever():
    docs = []

    for file in st.session_state.filenames:
        loader = TextLoader(os.path.join(DOCS_DIR, file))
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

    system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone \
    question which can be understood without the chat history. Do NOT answer \
    the question, just reformulate it if needed."""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return prompt_template | ChatOpenAI() | StrOutputParser()


def configure_chatbot():
    st.write("Reading configuration file...")
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        st.session_state.config = yaml.safe_load(file)

    st.write("Initializing message history...")
    st.session_state.history = StreamlitChatMessageHistory(key="messages")

    st.write("Reading documents...")
    st.session_state.filenames = [
        file
        for file in os.listdir(DOCS_DIR)
        if file.split(".")[-1] in ["txt", "md"]
    ]

    st.write("Connecting to user feedback database...")
    if not firebase_admin._apps:
        cert = dict(st.secrets["FIREBASE_AUTH"])
        cred = credentials.Certificate(cert)
        firebase_admin.initialize_app(cred)
        st.session_state.db = firestore.client()

    st.write("Finishing chatbot configuration...")
    st.session_state.configured = True
    st.rerun()


def get_messages_dump():
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
    st.title("Helpful Information")
    st.write(st.session_state.config["app_description"])

    with st.expander("Uploaded files"):
        st.markdown(
            "\n".join(
                f"- **{filename}**" for filename in st.session_state.filenames
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
        data=get_messages_dump(),
        file_name=f"chat-history-{time.time()}.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )


def setup_feedback_form():
    st.title("Feedback")
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


def setup_chat():
    if not st.session_state.messages:
        st.session_state.history.add_ai_message("How can I help you?")

    view_messages = st.expander("View the message contents in session state")

    retriever = setup_retriever()

    system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(
                lambda docs: "\n\n".join(
                    doc.page_content for doc in docs["context"]
                )
            )
        )
        | prompt_template
        | ChatOpenAI()
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "context": prompt_contextualizer | retriever,
        }
    ).assign(answer=rag_chain_from_docs)

    for message in st.session_state.messages:
        st.chat_message(message.type).write(message.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        response = rag_chain_with_source.invoke(
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

    with view_messages:
        view_messages.json(st.session_state.messages)


if __name__ == "__main__":
    st.set_page_config(page_title="LangChain Q&A with RAG", page_icon="📖")
    st.title("📖 LangChain Q&A with RAG")

    st.session_state.setdefault("configured", False)

    if not st.session_state.configured:
        configure_chatbot()
    else:
        setup_sidebar()
        setup_chat()
