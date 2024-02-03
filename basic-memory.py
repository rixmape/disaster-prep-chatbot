# fmt: off

import os
import tempfile

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.get("LANGCHAIN_ENDPOINT", "https://api.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "default")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

# fmt: on

def setup_file_uploader():
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()

    return uploaded_files


def setup_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
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
    # Do not contextualize the question if there is no history
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

    return (
        prompt_template | ChatOpenAI() | StrOutputParser()
    )


st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

uploaded_files = setup_file_uploader()
retriever = setup_retriever(uploaded_files)

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

rag_chain = (
    RunnablePassthrough.assign(context=prompt_contextualizer | retriever)
    | prompt_template
    | ChatOpenAI()
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    msgs.add_user_message(prompt)
    st.chat_message("human").write(prompt)

    response = rag_chain.invoke({"question": prompt, "history": msgs.messages})

    msgs.add_ai_message(response.content)
    st.chat_message("ai").write(response.content)

with view_messages:
    view_messages.json(st.session_state.langchain_messages)
