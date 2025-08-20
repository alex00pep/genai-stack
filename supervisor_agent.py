from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


from agentic_rag import graph as rag_agent
from math_agent import math_agent
from research_agent import research_agent
from langgraph_printer import pretty_print_messages
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# llm_model = "llama3.2:3b-instruct-fp16"

# For thinking
llm_model = "qwen2.5:14b"
llm = ChatOllama(model=llm_model)
supervisor = create_supervisor(
    model=llm,
    agents=[rag_agent, research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "- a RAG agent. Assign AI and ML related tasks to this agent, as it contains documents related to AI agents, prompt engineering, and adversarial attacks.\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()


if __name__ == "__main__":
    # for chunk in supervisor.stream(
    #     {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
    #             }
    #         ]
    #     },
    # ):
    #     pretty_print_messages(chunk, last_message=True)

    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find information about Chain of Thought in AI and ML?",
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)

        # final_message_history = chunk["supervisor"]["messages"]
