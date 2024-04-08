import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


class DocumentManager:
    def __init__(self, configuration):
        self.config = configuration

        self.embedding_model = self.initialize_embedding_model()
        self.vectorstore = self.initialize_vectorstore()
        self.retriever = self.get_retriever()

    def initialize_embedding_model(self):
        return OpenAIEmbeddings(
            model=self.config["parameters"]["embedding_model"],
            dimensions=self.config["parameters"]["embedding_dimensions"],
        )

    def initialize_vectorstore(self):
        return PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=self.embedding_model,
        )

    def get_retriever(self):
        return self.vectorstore.as_retriever(
            search_type=self.config["parameters"]["search_type"],
            search_kwargs={
                "k": self.config["parameters"]["docs_to_use"],
                "fetch_k": self.config["parameters"]["docs_to_process"],
            },
        )

    def get_documents(self, path="documents", limit=None):
        docs = []
        valid_file_types = self.config["parameters"]["allowed_file_types"]
        for filename in os.listdir(path)[:limit]:
            file_type = filename.split(".")[-1]
            if file_type not in valid_file_types:
                continue
            doc_loader = TextLoader(
                os.path.join(path, filename),
                encoding="utf-8",
            )
            docs.extend(doc_loader.load())
        return docs

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["parameters"]["embedding_chunk_size"],
            chunk_overlap=self.config["parameters"]["embedding_chunk_overlap"],
        )
        return text_splitter.split_documents(documents)
