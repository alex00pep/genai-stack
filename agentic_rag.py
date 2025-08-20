import os
import json
import operator
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Sequence, TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langgraph.graph import END
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv(".env")


from retrievers import ai_topics_retriever as aitr, st_overflow_retriever as stovr
from chains import load_llm
from tools import web_search_tool
from populate_stores import populate

### LLM
qdrant_url = os.getenv("QDRANT_URL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

llm = load_llm(llm_name, config={"ollama_base_url": ollama_base_url})
llm_json_mode = ChatOllama(
    model=llm_name, temperature=0, format="json", base_url=ollama_base_url
)

### Load AI topics vector store
vectorstore = populate()

### Router


# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The first vectorstore contains documents related to AI agents, prompt engineering, and adversarial attacks.
The second vectorstore questions and answers from software developers posted on Stack Overflow.

Use the adequate vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'ai-vectorstore' or 'stackoverflow-vectorstore' depending on the question."""


# %%
### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question} \n\n. 

Think carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


# %%
### Generate

# Prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
### Hallucination Grader

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""


# %%
### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


# %% [markdown]
# # LangGraph - State
#
# Graph state. The graph state schema contains keys that we want to:
#
# - Pass to each node in our graph
# - Optionally, modify in each node of our graph

# %%


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents
    messages: Annotated[Sequence[BaseMessage], add_messages]


# %% [markdown]
# # LangGraph - nodes and edges for LangChain's langgraph
# Each node in our graph is simply a function that:
#
# (1) Take state as an input. See conceptual notes aboute Graph state [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
#
# (2) Modifies state
#
# (3) Write the modified state to the state schema (dict)
#
# See conceptual docs [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes).
#
# Each edge routes between nodes in the graph.
#
# See conceptual docs [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges).

# %%


### Nodes
def retrieve_ai_docs(state):
    """
    Retrieve documents from AI topics in vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE from AI docs---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    instance = vectorstore.as_retriever(k=4)
    documents = instance.invoke(question)
    return {"documents": documents}


def retrieve_so_docs(state):
    """
    Retrieve documents from Stackoverflow documents in vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE from SO docs---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    instance = stovr()
    documents = instance.invoke(question)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENTS RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    from langchain.agents import initialize_agent, AgentType

    tools = [web_search_tool]
    # Agent with web search tool
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
    #                          handle_parsing_errors=True)
    # web_results = agent.invoke({"input": question})
    web_results = web_search_tool.invoke({"query": question})
    print("---WEB SEARCH RESULTS---")
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


def parse_input(state: GraphState, config: RunnableConfig):
    """
    Parse input to extract question and messages

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with question and messages
    """
    print("---PARSE INPUT---")
    if "messages" in state:
        # Extract question from messages
        question = state["messages"][0].content
        messages = state["messages"]
    else:
        # Extract question from input
        question = state["question"]
        messages = []

    return {"question": question, "messages": messages}


### Edges


def route_question(state: GraphState, config: RunnableConfig):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")

    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "ai-vectorstore":
        print("---ROUTE QUESTION TO AGENTIC RAG with ---")
        return "ai-vectorstore"
    elif source == "stackoverflow-vectorstore":
        print("---ROUTE QUESTION TO GraphRAG with Neo4j as Vector Store---")
        return "stackoverflow-vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            state["documents"] = []
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        state["documents"] = []
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"


# %% [markdown]
# # Control Flow

# %%

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("parse_input", parse_input)  # Parse input
workflow.add_node("websearch", web_search)  # web search
workflow.add_node(
    "retrieve_ai_docs", retrieve_ai_docs
)  # retriever for AI topics in its vector store
workflow.add_node(
    "retrieve_so_docs", retrieve_so_docs
)  # retriever for StackOverflow topics in its vector store
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.add_conditional_edges(
    "parse_input",
    route_question,
    {
        "websearch": "websearch",
        "ai-vectorstore": "retrieve_ai_docs",
        "stackoverflow-vectorstore": "retrieve_so_docs",
    },
)

workflow.set_entry_point("parse_input")  # Start node
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve_ai_docs", "grade_documents")
workflow.add_edge("retrieve_so_docs", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Compile the state graph into a runnable object
memory = MemorySaver()
graph = workflow.compile(name="agentic_rag", checkpointer=memory)


def call_rag_agent_chain(
    user_input: str,
    config: dict,
    language: str = "English",
    stream_mode: str = "values",
):
    if stream_mode == "messages":
        print("---STREAMING tokens---")
        for chunk, metadata in graph.stream(
            {"question": user_input, "language": language},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                print(chunk.content, end=" ")
    else:
        print("---STREAMING messages---")
        user_input = {"question": user_input, "language": language}
        for step in graph.stream(input=user_input, config=config, stream_mode="values"):
            # for value in step.values():
            #     print("Assistant:", value["messages"][-1].content)
            print(step)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc4576"}}
    query = "What is the meaning of life."
    language = "English"

    call_rag_agent_chain(query, config)
