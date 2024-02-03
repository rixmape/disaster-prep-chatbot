# fmt: off

import os
import tempfile

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# fmt: on


def configure_retriever(uploaded_files):
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


def question_getter(input):
    return input["question"]


st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Get an OpenAI API Key before continuing
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets.OPENAI_API_KEY
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Set up the LangChain, passing in Message History
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
    RunnablePassthrough.assign(context=question_getter | retriever)
    | prompt_template
    | ChatOpenAI(api_key=openai_api_key)
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
