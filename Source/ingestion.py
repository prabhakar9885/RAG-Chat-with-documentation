import os

from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader(path="../Data")
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} docs")

    # Send at 4-5 contexts in the LLM query.
    #    Assuming, we reserve 2k tokens for the context
    #    If we are sending 4 contexts in the LLM query, then the chunk size should be 2k/4 = 500 tokens.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)
    for doc in docs:
        doc.metadata["source"] = doc.metadata["source"].replace("../Data/langchain-docs", "https:/")

    print(f"Uploading the docs to Pinecode...")
    PineconeVectorStore.from_documents(documents=docs, embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Done")


if __name__ == '__main__':
    ingest_docs()
