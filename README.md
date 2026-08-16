# Agentic AI Programming Test — Two-Agent RAG System

A simple multi-agent system built with **LangChain / LangGraph** and **Google Gemini**
(free tier). A **Data Retriever** agent searches a local knowledge base using a
custom retrieval tool, and a **Report Generator** agent synthesizes the retrieved
snippets into a final, well-formatted answer.

## Architecture

```
User query
    │
    ▼
┌─────────────────────┐   raw snippets   ┌─────────────────────┐
│   Data Retriever    │ ───────────────▶ │  Report Generator   │
│  (ReAct agent with  │                  │  (LLM synthesizer,  │
│   search tool)      │                  │   no tools)         │
└─────────┬───────────┘                  └─────────┬───────────┘
          │ tool call                              │
          ▼                                        ▼
  knowledge_base.txt                        Final answer
```

Orchestration is a sequential **LangGraph `StateGraph`**:
`START → retrieve → generate → END`, with a shared typed state
(`query`, `snippets`, `answer`) flowing between the nodes.

- **Data Retriever** — a ReAct agent (`create_agent`) configured with the
  custom `search_knowledge_base` tool. Its instructions force it to always call
  the tool and return raw snippets without answering the question itself.
- **Report Generator** — a plain LLM call with a synthesizer system prompt. It is
  instructed to use *only* the provided snippets, remove redundancy, and say so
  explicitly when the knowledge base has no relevant information (prevents
  hallucination).

## RAG mechanism (custom tool)

`tools.py → search_knowledge_base(query)`:

1. Loads `knowledge_base.txt` and splits it into paragraph chunks.
2. **Semantic search** — embeds all chunks once with Gemini embeddings
   (cached with `lru_cache`), embeds the query, and ranks chunks by cosine
   similarity, returning the top-k.
3. **Keyword fallback** — if the embedding API is unavailable (quota/network),
   falls back to a stopword-filtered keyword-overlap score so the pipeline
   still works end to end.
4. Returns raw text chunks separated by `---`, or
   `NO_RELEVANT_INFORMATION_FOUND` when nothing matches.

## Project structure

```
├── knowledge_base.txt   # sample company policies (the knowledge base)
├── tools.py             # custom retrieval tool (semantic + keyword fallback)
├── agents.py            # agent definitions and system prompts
├── graph.py             # LangGraph sequential orchestration
├── main.py              # entry point with sample queries
├── requirements.txt
├── .env.example
└── screenshots/         # output screenshots for sample queries
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then put your key in .env
```

Get a free Gemini API key at https://aistudio.google.com/apikey and set it in
`.env` as `GOOGLE_API_KEY`.

## Run

```bash
python main.py                                        # built-in sample queries
python main.py "What is the policy on international travel?"
```

Each run prints both the raw snippets returned by the Data Retriever and the
final synthesized answer from the Report Generator.

## Sample queries covered

1. `What is the policy on international travel?` — direct hit on one chunk.
2. `How many days of annual leave do employees get, and can unused days be carried over?` — multiple facts within a chunk.
3. `What are the rules for working remotely?` — paraphrased query (semantic match).
4. `What is the company's policy on pets in the office?` — **not** in the KB; the
   system correctly reports that the information is unavailable instead of
   hallucinating.

## Possible extensions (out of scope for this test)

- Replace the in-memory chunk list with a vector store (FAISS / Chroma) for
  larger corpora, plus sentence-level chunking with overlap.
- Add a reranker step and citation of source chunks in the final answer.
- Expose the pipeline behind an API (FastAPI) with streaming responses.
