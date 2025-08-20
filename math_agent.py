from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_printer import pretty_print_messages

# Load LLMs
load_dotenv()
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


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)


if __name__ == "__main__":
    # Example usage

    for chunk in math_agent.stream(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 7"}]}
    ):
        pretty_print_messages(chunk)
