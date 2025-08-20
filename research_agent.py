from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools import web_search_tool
from langgraph_printer import pretty_print_messages
from langgraph.checkpoint.memory import MemorySaver


base_url = "http://192.168.0.108:11434"

# router_llm = "llama3.2:3b-instruct-fp16"
# router_llm = "llama3-groq-tool-use:latest"
router_llm = "qwen2.5:14b"
llm = ChatOllama(
    model=router_llm,
    temperature=0,
    base_url=base_url,
    validate_model_on_init=True,
)

# Add memory
memory = MemorySaver()

research_agent = create_react_agent(
    model=llm,
    tools=[web_search_tool],
    checkpointer=memory,
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


if __name__ == "__main__":
    # Example usage

    for chunk in research_agent.stream(
        {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
    ):
        pretty_print_messages(chunk)
