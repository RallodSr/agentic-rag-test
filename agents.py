"""Agent definitions: Data Retriever (RAG) and Report Generator."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import search_knowledge_base

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

RETRIEVER_SYSTEM_PROMPT = (
    "You are the Data Retriever, an expert in information retrieval.\n"
    "Your ONLY job is to find information in the knowledge base:\n"
    "1. ALWAYS call the search_knowledge_base tool with the user's query.\n"
    "2. Return the retrieved snippets EXACTLY as the tool provides them,\n"
    "   without summarizing, rewriting, or adding your own knowledge.\n"
    "3. Never answer the user's question yourself.\n"
    "4. If the tool returns NO_RELEVANT_INFORMATION_FOUND, return exactly\n"
    "   that string."
)

GENERATOR_SYSTEM_PROMPT = (
    "You are the Report Generator, an expert writer and synthesizer.\n"
    "You receive raw text snippets retrieved from a knowledge base and the\n"
    "user's original question. Write the final answer for the user:\n"
    "- Use ONLY the information contained in the snippets. Do not invent\n"
    "  facts or rely on outside knowledge.\n"
    "- Remove redundancy: if snippets repeat the same fact, state it once.\n"
    "- Format the answer clearly (short paragraphs or bullet points).\n"
    "- If the snippets are 'NO_RELEVANT_INFORMATION_FOUND' or do not answer\n"
    "  the question, say plainly that the knowledge base does not contain\n"
    "  this information."
)


def build_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model. Temperature 0 keeps answers grounded."""
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)


def build_retriever_agent():
    """ReAct-style agent configured with the custom retrieval tool."""
    return create_react_agent(
        model=build_llm(),
        tools=[search_knowledge_base],
        prompt=RETRIEVER_SYSTEM_PROMPT,
    )


def run_report_generator(query: str, snippets: str) -> str:
    """Synthesize the final answer from the retrieved snippets."""
    llm = build_llm(temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{query}\n\n"
                    f"Retrieved snippets:\n{snippets}"
                )
            ),
        ]
    )
    return response.text
