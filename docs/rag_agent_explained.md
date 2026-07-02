# RAG Agent — Complete Code Explanation

> File: `backend/rag_agent.py`  
> Phase 7 (Clinical Intelligence) inside Phase 3 (MLOps Backend)

---

## What is a RAG Agent?

**RAG = Retrieval-Augmented Generation.**  
Instead of asking an LLM to answer purely from its training data, you first **retrieve relevant documents** from your own knowledge base, then **give those documents to the LLM** so it can reason over real, up-to-date, domain-specific content.

This file adds a layer on top of that: a **LangChain Agent** that can *decide which tool to call* rather than blindly retrieving and dumping text.

```
Plain LLM:   Question → LLM → Answer
RAG:         Question → Retrieve docs → LLM(docs + question) → Answer
RAG Agent:   Question → Agent thinks → calls tool → observes → thinks → ... → Answer
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    query_agent_with_trace()              │
│                                                         │
│  ┌──────────┐    ┌─────────────────────────────────┐   │
│  │   LLM    │◄──►│       LangChain ReAct Agent      │   │
│  │ (vLLM or │    │  Thought → Action → Observation  │   │
│  │  GPT-2)  │    └──────────┬──────────────────────┘   │
│  └──────────┘               │ calls one of 3 tools      │
│                    ┌────────┼────────┐                  │
│                    ▼        ▼        ▼                  │
│             search_   explain_  summarise_              │
│             similar_  finding   report                  │
│             cases                                       │
│                    │                                    │
│                    ▼                                    │
│         ┌──────────────────┐                           │
│         │   Vector Store   │                           │
│         │ ChromaDB (prim.) │                           │
│         │ FAISS (fallback) │                           │
│         └──────────────────┘                           │
│                    ▲                                    │
│                    │ encodes text                       │
│             ClinicalBERT                               │
└─────────────────────────────────────────────────────────┘
```

---

## Section-by-Section Code Walkthrough

---

### 1. Dependency Guards (try/except imports)

```python
try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from langchain.agents import AgentExecutor, create_react_agent
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
```

**Why this pattern?**  
Every heavy dependency is wrapped in `try/except`. This means the entire API server still starts even if you haven't installed ChromaDB, FAISS, or LangChain. The code degrades gracefully:

| What's missing | What still works |
|---|---|
| `chromadb` | Falls back to FAISS |
| `faiss` | Falls back to plain text search |
| `langchain` | Falls back to raw FAISS results |
| `transformers` | No vector search at all |

---

### 2. ClinicalBERT Encoder

```python
CLINICAL_BERT_MODEL = os.getenv("CLINICAL_BERT_MODEL", "medicalai/ClinicalBERT")

_tokenizer = None
_bert_model = None

def _get_clinical_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:                          # lazy load — only once
        _tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
        _bert_model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL)
        _bert_model.eval()
    return _tokenizer, _bert_model
```

**What is ClinicalBERT?**  
A BERT model fine-tuned on clinical notes (MIMIC-III). It understands medical vocabulary far better than general BERT — words like "cardiomegaly", "costophrenic", "atelectasis" have meaningful embeddings.

```python
def encode_texts(texts: list[str]) -> np.ndarray:
    tokenizer, model = _get_clinical_bert()
    embeddings = []
    with torch.no_grad():                           # no gradients needed (inference only)
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            outputs = model(**inputs)
            # Mean-pool the last hidden state → one vector per text
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)
```

**What does mean-pooling do?**  
BERT outputs one vector per token (e.g. 512 vectors for a 512-token report).  
Mean-pooling averages them into **one 768-dimensional vector** that represents the whole sentence.

```
"Cardiomegaly noted. Bilateral pleural effusion."
    ↓ tokenize
["Card", "##io", "##meg", "##aly", "noted", ...]   ← 512 tokens max
    ↓ BERT
[[0.2, -0.1, ...], [0.5, 0.3, ...], ...]           ← 512 × 768 matrix
    ↓ mean pool
[0.35, 0.1, ...]                                   ← 1 × 768 vector
```

Two similar reports will have vectors that are **close together** in 768-dimensional space. That's what makes semantic search work.

---

### 3. ChromaDB Vector Store (Primary)

```python
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

def _get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, ...)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="radiology_reports",
            metadata={"hnsw:space": "cosine"},     # cosine similarity search
        )
    return _chroma_client, _chroma_collection
```

**ChromaDB key concepts:**

| Concept | Meaning |
|---|---|
| `PersistentClient` | Data is saved to disk at `chroma_store/` — survives restarts |
| `collection` | Like a table — groups related documents together |
| `hnsw:space: cosine` | Uses cosine similarity instead of Euclidean distance |
| `metadata` | Extra fields stored per document (age, view, modality) — filterable |

**Cosine similarity vs L2 distance:**
- L2: measures straight-line distance between vectors (sensitive to vector magnitude)
- Cosine: measures the *angle* between vectors (ignores magnitude — better for text)

```
Score 1.0 = identical meaning
Score 0.5 = somewhat similar
Score 0.0 = unrelated
```

---

### 4. FAISS Fallback

```python
_faiss_index = None
_indexed_reports: list[str] = []

def _load_faiss():
    if os.path.exists(FAISS_INDEX_FILE):
        _faiss_index = faiss.read_index(FAISS_INDEX_FILE)   # load from disk
        with open(FAISS_TEXTS_FILE) as f:
            _indexed_reports = json.load(f)                  # load raw texts

def _save_faiss():
    faiss.write_index(_faiss_index, FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "w") as f:
        json.dump(_indexed_reports, f)
```

**Why FAISS needs manual save/load (unlike ChromaDB):**  
FAISS is a pure in-memory library — it only does fast vector math. It has no database layer. So we manually persist:
- `reports.index` — the FAISS binary index (vectors only)
- `reports.json` — the original text strings (FAISS stores no text, just numbers)

**At startup**, if ChromaDB is not installed, FAISS tries to load any previously saved index from disk so you don't lose your data between restarts.

---

### 5. `build_index()` — Adding Reports

```python
def build_index(reports: list[str], metadatas: list[dict] = None):
    embeddings = encode_texts(reports)          # ClinicalBERT → vectors

    if _CHROMA_AVAILABLE:
        _, collection = _get_chroma()
        start_id = collection.count()           # offset so IDs don't collide
        ids = [str(start_id + i) for i in range(len(reports))]
        collection.add(
            ids=ids,
            documents=reports,                  # raw text stored alongside vector
            embeddings=embeddings.tolist(),
            metadatas=metadatas,                # e.g. {"view": "PA", "age": 65}
        )
        return

    if _FAISS_AVAILABLE:
        index = faiss.IndexFlatL2(embeddings.shape[1])   # L2 flat index
        index.add(embeddings)
        _faiss_index = index
        _indexed_reports = reports
        _save_faiss()
```

**The flow:**
```
["Report 1 text", "Report 2 text", ...]
        ↓ encode_texts()
[[0.2, 0.5, ...], [0.8, 0.1, ...], ...]   ← float32 vectors
        ↓ ChromaDB / FAISS
Stored to disk with original text + metadata
```

---

### 6. `retrieve_similar_reports()` — Semantic Search

```python
def retrieve_similar_reports(query, k=3, filters=None):
    q_emb = encode_texts([query])               # encode the query the same way

    if _CHROMA_AVAILABLE:
        results = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=k,
            where=filters,                      # e.g. {"view": "PA"}
            include=["documents", "distances", "metadatas"],
        )
        # Convert cosine distance → similarity score
        score = round(1 - dist / 2, 4)         # dist=0 → score=1.0
        ...

    if _FAISS_AVAILABLE:
        _, indices = _faiss_index.search(q_emb, k)
        return [{"text": _indexed_reports[i], ...} for i in indices[0]]
```

**How similarity search works:**
```
Query: "elderly patient with enlarged heart"
        ↓ encode → [0.3, 0.7, ...]
        
Distance to each stored report:
  Report A "cardiomegaly, CHF"     → dist 0.12  → score 0.94  ✓ very similar
  Report B "pneumothorax, young"   → dist 0.88  → score 0.56  ✗ not similar
  Report C "cardiac enlargement"   → dist 0.18  → score 0.91  ✓ similar

Returns top-k by score.
```

**ChromaDB filter example:**
```python
retrieve_similar_reports(
    "heart failure",
    filters={"view": "PA", "age": {"$gte": 60}}
)
# Only searches reports tagged as PA-view AND age ≥ 60
```

---

### 7. vLLM Client (`_get_llm()`)

```python
def _get_llm():
    if VLLM_BASE_URL:                           # production: vLLM server
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=f"{VLLM_BASE_URL}/v1",    # vLLM exposes OpenAI-compatible API
            api_key="EMPTY",
            model=VLLM_MODEL,
            temperature=0.2,                    # low = more deterministic
        )

    # development fallback: tiny GPT-2 locally
    from langchain_huggingface import HuggingFacePipeline
    pipe = hf_pipeline("text-generation", model="sshleifer/tiny-gpt2", ...)
    return HuggingFacePipeline(pipeline=pipe)
```

**vLLM vs tiny-gpt2:**

| | vLLM (production) | tiny-gpt2 (dev) |
|---|---|---|
| Quality | High (Mistral 7B) | Very low (2M params) |
| Speed | Fast (continuous batching) | Slow |
| Setup | Needs GPU server | Runs on CPU |
| Purpose | Real clinical answers | Testing the agent loop works |

**Why `temperature=0.2`?**  
Lower temperature = the model picks higher-probability tokens = more consistent, factual answers. Higher temperature = more creative but also more hallucinations — bad for medical use.

---

### 8. The ReAct Prompt

```python
_REACT_PROMPT = PromptTemplate.from_template("""
You are a clinical AI assistant specialising in chest X-ray radiology.

{tools}

Use the following format:
Question: the input question you must answer
Thought: your reasoning
Action: the tool name
Action Input: the input to the tool
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}
""")
```

**What is ReAct?**  
ReAct (Reason + Act) is a prompting pattern where the LLM alternates between:
- **Thought** — "I need to search for similar cases first"
- **Action** — calls a tool
- **Observation** — reads the tool's result
- repeats until it writes **Final Answer**

The `{agent_scratchpad}` placeholder is filled in by LangChain with the growing history of Thoughts/Actions/Observations from the current session.

**Why this works:**  
The LLM was trained on text that contains reasoning patterns. By structuring the prompt this way, the model follows the pattern and produces valid tool calls.

---

### 9. The Three Agent Tools

```python
def _build_agent_tools(report_context: str):
```

This function creates three tools, each a plain Python function wrapped in LangChain's `Tool` class.

---

#### Tool 1: `search_similar_cases`

```python
def search_similar_cases(query: str) -> str:
    results = retrieve_similar_reports(query, k=3)
    # formats results as text the LLM can read
    return "\n---\n".join([f"{r['text']} (similarity: {r['score']})" for r in results])
```

- **Input:** any clinical description string
- **Does:** calls `retrieve_similar_reports()` → FAISS/ChromaDB semantic search
- **Returns:** the top-3 similar reports as a formatted string
- **When agent uses it:** "I need real examples from the knowledge base"

---

#### Tool 2: `explain_finding`

```python
def explain_finding(finding: str) -> str:
    explanations = {
        "cardiomegaly": "Enlargement of the cardiac silhouette...",
        "pleural effusion": "Fluid in the pleural space...",
        "pneumothorax": "Air in the pleural space...",
        ...
    }
    key = finding.lower().strip()
    for k, v in explanations.items():
        if k in key:
            return v
    return f"No pre-loaded explanation for '{finding}'."
```

- **Input:** a finding name like `"cardiomegaly"`
- **Does:** dictionary lookup for clinical definitions
- **Returns:** pathophysiology + imaging features
- **When agent uses it:** "I need to explain what this finding means clinically"

> The dictionary is intentionally small here — in production you'd replace this with another vector search over a clinical textbook corpus.

---

#### Tool 3: `summarise_report`

```python
def summarise_report(text: str) -> str:
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return ". ".join(sentences[:2]) + "."
```

- **Input:** full report text
- **Does:** takes first 2 meaningful sentences
- **Returns:** a short summary
- **When agent uses it:** "The report context is too long — let me condense it first"

> This is a simple heuristic. In production, you'd call the LLM itself to summarise, or use an extractive summarisation model.

---

### 10. `query_agent_with_trace()` — The Main Entry Point

```python
def query_agent_with_trace(question, report_context="") -> dict:
```

This is the function that orchestrates everything. Here's the full flow:

```
query_agent_with_trace("What does cardiomegaly mean in elderly?")
│
├─ LangChain not available?
│     └─ retrieve_similar_reports() → return raw results (no reasoning)
│
├─ No LLM available?
│     └─ return error message
│
└─ Build agent + executor
      │
      └─ executor.invoke({"input": question})
            │
            ├─ Step 1: Agent thinks → calls search_similar_cases("cardiomegaly elderly")
            │          Tool returns 3 reports from FAISS
            │
            ├─ Step 2: Agent thinks → calls explain_finding("cardiomegaly")
            │          Tool returns clinical definition
            │
            └─ Agent writes Final Answer combining both observations
```

**Capturing the trace:**

```python
executor = AgentExecutor(
    ...
    return_intermediate_steps=True,   # ← this is the key flag
)
result = executor.invoke({"input": question})

# result["intermediate_steps"] = list of (AgentAction, observation) tuples
for action, observation in result["intermediate_steps"]:
    steps.append({
        "thought":     action.log,          # LLM's reasoning text
        "tool":        action.tool,         # tool name
        "tool_input":  action.tool_input,   # what was passed to the tool
        "observation": observation,         # what the tool returned
    })
```

Without `return_intermediate_steps=True`, you only get the final answer. With it, you get the full chain of reasoning — which is what the Gradio UI displays in the **Reasoning Trace** box.

---

### 11. `query_agent()` — Backward-Compatible Wrapper

```python
def query_agent(question, report_context="") -> str:
    return query_agent_with_trace(question, report_context)["answer"]
```

This exists so that any code that was already calling `query_agent()` keeps working. It just calls the new function and discards the trace.

---

## Data Flow Diagram (End to End)

```
POST /agent/query
  {"question": "cardiomegaly in elderly?", "return_trace": true}
          │
          ▼
    query_agent_with_trace()
          │
          ├──► _get_llm()  →  vLLM / HuggingFace pipeline
          │
          ├──► _build_agent_tools()
          │       ├── search_similar_cases  ──► retrieve_similar_reports()
          │       │                                   └──► encode_texts() [ClinicalBERT]
          │       │                                   └──► ChromaDB / FAISS search
          │       ├── explain_finding       ──► dictionary lookup
          │       └── summarise_report      ──► sentence splitter
          │
          └──► AgentExecutor.invoke()
                  │
                  ├── Step 1: Thought + Action + Observation
                  ├── Step 2: Thought + Action + Observation
                  └── Final Answer
          │
          ▼
    {
      "answer": "Cardiomegaly in elderly patients...",
      "langchain_active": true,
      "steps": [
        {"thought": "...", "tool": "search_similar_cases", ...},
        {"thought": "...", "tool": "explain_finding", ...}
      ]
    }
```

---

## Fallback Chain Summary

```
Is chromadb installed?
  YES → use ChromaDB (persistent, filterable, cosine similarity)
  NO  →
    Is faiss installed?
      YES → use FAISS (in-memory, L2 distance, manual save/load)
      NO  → return []

Is langchain installed?
  YES →
    Is an LLM available? (vLLM or transformers)
      YES → run full ReAct agent loop
      NO  → return error
  NO  → return raw FAISS/ChromaDB results directly
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Lazy loading of ClinicalBERT | 400MB model — only load when first needed |
| `try/except` on every import | Server starts even in minimal environments |
| `return_intermediate_steps=True` | Lets the UI show the agent's reasoning chain |
| `max_iterations=5` | Prevents infinite loops if the LLM gets stuck |
| `handle_parsing_errors=True` | If the LLM outputs malformed tool calls, agent retries |
| `temperature=0.2` | Favours factual, consistent medical answers |
| ChromaDB cosine space | Cosine similarity is more robust for variable-length text |
| Separate `query_agent()` wrapper | Preserves backward compatibility with existing callers |
