import os
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from chains import load_embedding_model
from langchain_community.vectorstores import SKLearnVectorStore


sto_store_url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
# Remapping for Langchain Neo4j integration
# os.environ["NEO4J_URL"] = ai_store_url
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
qdrant_url = os.getenv("QDRANT_URL")

# Load embeddings for similarity search
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}
)


def st_overflow_retriever():
    """Get the AI vector store."""

    # Vector + Knowledge Graph response
    st_overflow_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=sto_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
    WITH node AS question, score AS similarity
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..2] as answers
        RETURN reduce(str='', answer IN answers | str + 
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    } 
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n' 
        + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    ORDER BY similarity ASC // so that best answers are the last
    """,
    )

    return st_overflow_store.as_retriever(search_kwargs={"k": 3})


def ai_topics_retriever():
    # Create retriever
    # Qdrant vector-only store for AI topics.
    ai_topics_store = Qdrant.from_existing_collection(
        path=qdrant_url,
        collection_name="ai-agents-collection",
        embedding=embeddings,
    )

    return ai_topics_store.as_retriever(k=3)
