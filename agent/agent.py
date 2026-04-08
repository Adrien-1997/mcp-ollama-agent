"""LangChain ReAct agent wired to Ollama + MCP tools."""

import logging
from functools import lru_cache

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.mcp_adapter import load_mcp_tools
from agent.rag import similarity_search

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful local AI assistant with access to tools.

Available tools (use ONLY when the question genuinely requires them):
{tools}

Rules:
- For greetings, conversational messages, or questions you can answer from general knowledge, go DIRECTLY to Final Answer. Do NOT use any tool.
- Only use a tool when you need live information (web search), file access, or code execution.
- Only pass arguments that the tool explicitly accepts. Do not invent extra fields.

Use this exact format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (JSON object with only valid fields)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Context from memory:
{context}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


async def build_agent() -> AgentExecutor:
    tools = await load_mcp_tools()

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        timeout=120,  # model cold-start can take 20-30s on CPU
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )


async def run_agent(query: str, agent_executor: AgentExecutor) -> dict:
    # Retrieve relevant context from vector store
    try:
        docs = similarity_search(query, k=3)
        context = "\n".join(d.page_content for d in docs) if docs else "No prior context."
    except Exception:
        context = "Vector store not available."

    result = await agent_executor.ainvoke({
        "input": query,
        "context": context,
    })

    return {
        "output": result.get("output", ""),
        "intermediate_steps": [
            {"action": str(step[0]), "observation": str(step[1])}
            for step in result.get("intermediate_steps", [])
        ],
    }
