import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class DocumentManager:
    def __init__(self, storage_bucket, document_directory="documents"):
        self.storage_bucket = storage_bucket
        self.document_directory = document_directory
        self.document_filenames = self.download_documents()
        self.retriever = self.initialize_retriever()

    def download_documents(self):
        os.makedirs(self.document_directory, exist_ok=True)
        storage_blobs = list(self.storage_bucket.list_blobs())
        document_filenames = []
        for blob in storage_blobs:
            filename = f"{self.document_directory}/{blob.name}"
            blob.download_to_filename(filename)
            document_filenames.append(filename)
        return document_filenames

    def initialize_retriever(self):
        documents = []
        for filename in self.document_filenames:
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
