import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def get_results_from_rag(query: str):
    embeddingz = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(embedding=embeddingz, index_name=os.environ.get('INDEX_NAME'))
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = Ollama(model="llama3.2:latest", temperature=0)
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)

    qa = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=stuff_docs_chain)
    res = qa.invoke(input={"input": query})
    return res


if __name__ == '__main__':
    res = get_results_from_rag(query="What is chunking?")
    print(res)
