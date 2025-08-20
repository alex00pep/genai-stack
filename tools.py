from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from langchain.agents import Tool

# Load LLMs
load_dotenv()
### Web search tool - Serperdev
search = GoogleSerperAPIWrapper()
web_search_tool = Tool(
    name="Intermediate Answer",
    func=search.run,
    description="useful for when you need to ask with search",
)
