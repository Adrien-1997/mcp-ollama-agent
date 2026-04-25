"""LangChain ReAct agent wired to Ollama + MCP tools."""

import logging
from functools import lru_cache

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from agent.config import settings
from agent.mcp_adapter import load_mcp_tools

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful local AI assistant with expertise in the local AI ecosystem. \
You are the model {ollama_model}, running locally via Ollama.
If anyone asks which model you are, what model is in use, or anything about your own identity, answer directly: you are {ollama_model}.

You have access to a knowledge base with documentation for these local AI tools:
Ollama, llama.cpp, LocalAI, Open WebUI, Jan, text-generation-webui, AnythingLLM, PrivateGPT, LiteLLM, GPT4All, Continue, Tabby.

Available tools (use ONLY when the question genuinely requires them):
{tools}

Rules:
- For greetings, conversational messages, or questions you can answer from general knowledge, go DIRECTLY to Final Answer. Do NOT use any tool.
- Use `query_knowledge_base` when the user asks about any local AI tool, how to install/configure/use it, or compares tools. Prefer this over web_search for ecosystem topics.
- Use `web_search` for current events, recent releases, or topics outside the knowledge base.
- Use `file_read` / `file_write` / `file_list` for workspace file operations.
- Use `code_exec` to run Python code or verify computations.
- Only pass arguments that the tool explicitly accepts. Do not invent extra fields.
- When answering from knowledge base results, cite the source document.

Use this exact format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (JSON object with only valid fields)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

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
    result = await agent_executor.ainvoke({
        "input": query,
        "ollama_model": settings.ollama_model,
    })

    return {
        "output": result.get("output", ""),
        "intermediate_steps": [
            {"action": str(step[0]), "observation": str(step[1])}
            for step in result.get("intermediate_steps", [])
        ],
    }
