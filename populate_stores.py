import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.schema import Document

from langchain_community.vectorstores import Qdrant

from chains import load_embedding_model

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
qdrant_url = os.getenv("QDRANT_URL")


def insert_ai_topics(
    doc_splits, embeddings, embeddings_store_url, username=None, password=None
):

    # Create a vector store in Qdrant
    # vectorstore = Qdrant.from_documents(
    #     documents=doc_splits,
    #     url=embeddings_store_url,
    #     embedding=embeddings,
    #     collection_name="ai-agents-collection",
    # )

    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    return vectorstore


def ai_topics_importer():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def ai_topics_splitter(docs_list: list[Document]):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


def populate():
    # Load documents from webbase loader and split them
    docs = ai_topics_importer()
    print("Loaded", len(docs), "documents")
    doc_splits = ai_topics_splitter(docs)
    # Save to vector store
    # Load embeddings for similarity search
    embeddings, dimension = load_embedding_model(
        embedding_model_name, config={"ollama_base_url": ollama_base_url}
    )
    # Create vector store from documents and embeddings
    vectorstore = insert_ai_topics(
        doc_splits=doc_splits, embeddings=embeddings, embeddings_store_url=qdrant_url
    )
    return vectorstore
