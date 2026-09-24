# 🩺 MedAssist — Multi-Agent Medical RAG Assistant

A multi-agent medical information assistant built with **LangGraph**, orchestrating
five cooperating agents over two retrieval-augmented knowledge bases (a disease
reference book and a medicine reference book). All language models run **locally
and for free via Ollama**, and every run is traced and benchmarked with **LangSmith**.

> ⚠️ **Educational project.** This assistant does not diagnose or prescribe. It is not
> a substitute for professional medical advice. A rule-based safety layer detects
> emergency language and redirects the user to real emergency services.

## Architecture

```
                         ┌─────────────────┐
              (missing)  │ Onboarding Agent│  extracts age/sex/symptoms/
            ┌───────────►│                 │  allergies/medications,
            │             └────────┬────────┘  asks for what's missing
            │                      │ (complete)
     START ─┤                      ▼
            │             ┌─────────────────┐
            └────────────►│Orchestrator Agent│  routes: disease / medicine /
       (emergency          └────────┬────────┘  both / direct / emergency
        keywords skip               │
        straight to Response)   ┌───┴────┐
                                 ▼        ▼
                      ┌─────────────┐ ┌──────────────┐
                      │Disease Agent│ │Medicine Agent │  each does RAG search
                      │  (RAG)      │►│  (RAG)        │  over its own Chroma
                      └──────┬──────┘ └───────┬───────┘  vector store
                             │                 │
                             ▼                 ▼
                          ┌───────────────────────┐
                          │    Response Agent      │  writes the final,
                          └───────────┬────────────┘  cited, disclaimed answer
                                      ▼
                                     END
```

- **Onboarding agent** — extracts patient info (age, sex, symptoms, allergies,
  current medications) turn by turn and asks a short follow-up question for
  whatever is still missing, instead of demanding a rigid form upfront.
- **Orchestrator agent** — reads the profile + latest message and routes the turn to
  the disease agent, the medicine agent, both, straight to the response agent
  (small talk), or flags an emergency.
- **Disease agent** — retrieves relevant passages from the *disease* vector store and
  lists conditions that could match the symptoms, plus red-flag warning signs.
- **Medicine agent** — retrieves relevant passages from the *medicine* vector store,
  checks for conflicts with the patient's allergies/current medications.
- **Response agent** — merges everything into one clear, cited, disclaimed answer.
- **Safety layer** — a regex-based emergency detector runs *before* the LLM
  router and always wins; emergencies get a fixed message, no model call.

State (profile, conversation, routing) is carried in a `LangGraph` `StateGraph`
and persisted per conversation with `MemorySaver`.

## Tech stack

| Purpose            | Tool                                   |
|---------------------|-----------------------------------------|
| Agent orchestration | LangGraph                                |
| LLM framework       | LangChain                                |
| Local LLMs / embeddings | Ollama (free, runs on your machine) |
| Vector store         | Chroma                                 |
| Observability        | LangSmith (traces, latency, evaluation) |
| UI                    | Streamlit (+ a terminal CLI)          |

## Project structure

```
medical-multi-agent/
├── app.py                     # Streamlit chat UI
├── requirements.txt
├── pyproject.toml
├── .env.example
├── data/
│   ├── disease_book/           # put your disease-reference PDF(s) here
│   └── medicine_book/          # put your medicine-reference PDF(s) here
├── vectorstore/                 # Chroma persistence (generated, gitignored)
├── scripts/
│   └── benchmark.py             # latency benchmark across sample questions
├── src/medassist/
│   ├── config.py                 # settings from .env
│   ├── llm.py                    # Ollama LLM / embeddings factories
│   ├── schemas.py                 # Pydantic structured-output schemas
│   ├── prompts.py                  # all agent prompts
│   ├── state.py                     # shared LangGraph state
│   ├── graph.py                      # graph definition & routing
│   ├── runner.py                      # runs one turn, collects timings
│   ├── cli.py                          # terminal chat loop
│   ├── agents/
│   │   ├── onboarding.py
│   │   ├── orchestrator.py
│   │   ├── disease.py
│   │   ├── medicine.py
│   │   └── response.py
│   ├── rag/
│   │   ├── ingest.py               # PDF -> chunks -> Chroma
│   │   └── retriever.py             # MMR retrieval + source formatting
│   └── utils/
│       ├── safety.py                 # emergency detection, disclaimers
│       ├── profile.py                 # profile merge / missing-fields logic
│       ├── messages.py                 # chat-history helpers
│       └── timing.py                    # per-node latency logging
└── tests/                                # pytest, LLM/retriever stubbed out
```

## Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Pull local models
```bash
ollama pull llama3.1:8b        # chat model (used by all agents by default)
ollama pull nomic-embed-text   # embedding model for RAG
```

### 3. Install
```bash
git clone https://github.com/<your-username>/medical-multi-agent.git
cd medical-multi-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `LANGSMITH_API_KEY` — get one free at [smith.langchain.com](https://smith.langchain.com)
  to enable tracing (optional but recommended).
- Change `OLLAMA_MODEL` / `EMBEDDING_MODEL` if you use different models.
- You can also override the model per agent, e.g. `ORCHESTRATOR_MODEL=llama3.2:3b`
  to use a smaller/faster model just for routing.

### 4. Add your reference books
Drop your PDFs into:
```
data/disease_book/your-disease-book.pdf
data/medicine_book/your-medicine-book.pdf
```

### 5. Build the vector stores
```bash
python -m medassist.rag.ingest          # both books
python -m medassist.rag.ingest disease  # only one, if you update just that book
```

### 6. Run

Terminal:
```bash
python -m medassist.cli
```

Web UI:
```bash
streamlit run app.py
```

## Benchmarking & monitoring

- **LangSmith** traces every agent call automatically (prompts, retrieved chunks,
  tokens, and per-node latency) once `LANGSMITH_TRACING=true` and an API key are
  set — check your project at [smith.langchain.com](https://smith.langchain.com).
- **Local benchmark** for a quick latency summary without opening LangSmith:
  ```bash
  python scripts/benchmark.py
  ```
  Saves per-turn timings to `results/benchmark.csv` and prints an average.

## Testing

Tests stub the LLM and vector store, so they run instantly with no Ollama or
GPU required — they check the graph wiring, routing rules, safety layer and
profile logic:
```bash
pytest -q
```

## Notes / possible extensions

- Swap `MemorySaver` for a persistent checkpointer (SQLite/Postgres) to keep
  conversations across restarts.
- Add a LangSmith **evaluation dataset** of sample Q&A pairs to score answer
  quality over time, not just latency.
- Add re-ranking (e.g. a cross-encoder) on top of the MMR retriever for higher
  precision on larger books.
- Swap models per agent to trade off speed vs. quality (e.g. a 3B model for
  routing, a larger model for the final response).

## Disclaimer

This project is for educational and portfolio purposes only. It does not
provide medical advice, diagnosis, or treatment, and must not be used as a
substitute for consulting a qualified healthcare professional.

## License

MIT — see [LICENSE](LICENSE).
