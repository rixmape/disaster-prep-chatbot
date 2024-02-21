# fmt: off

import os
from operator import itemgetter

import streamlit as st
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

# fmt: on


def setup_retriever(path="documents"):
    docs = []

    for file in os.listdir(path):
        loader = TextLoader(f"{path}/{file}")
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

    return prompt_template | ChatOpenAI() | StrOutputParser()


def setup_chat():
    # Set up the chat history
    msgs = StreamlitChatMessageHistory()
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

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

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        response = rag_chain_with_source.invoke(
            {
                "question": prompt,
                "history": msgs.messages,
            }
        )

        with st.chat_message("ai"):
            st.write(response.get("answer"))

            citation_container = st.expander(f"File Citations:", expanded=False)
            for citation in response.get("context"):
                source = citation.metadata.get("source")
                content = citation.page_content.replace("#", "")
                content = "\n".join(
                    [f"> {line}" for line in content.split("\n")]
                )
                citation_container.markdown(f"**{source}**\n{content}")

        msgs.add_user_message(prompt)
        msgs.add_ai_message(response.get("answer"))

    with view_messages:
        view_messages.json(st.session_state.langchain_messages)


if __name__ == "__main__":
    st.set_page_config(page_title="LangChain Q&A with RAG", page_icon="📖")
    st.title("📖 LangChain Q&A with RAG")

    setup_chat()
