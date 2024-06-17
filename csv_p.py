
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType

from langchain_community.llms.ollama import Ollama


llm = Ollama(model='llama3')
agent = create_csv_agent(
    llm,
    "titanic.csv",
    verbose = True,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    allow_dangerous_code = True
)

agent.run("who was passenger 1?")