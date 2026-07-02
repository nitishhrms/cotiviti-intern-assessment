from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("xray_api")

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — falling back to FAISS.")

try:
    import numpy as np
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False
    logger.warning("transformers not installed — text encoding unavailable.")

try:
    try:
        from langchain_classic.agents import AgentExecutor, create_react_agent
        from langchain_classic.tools import Tool
    except ImportError:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
    from langchain_core.prompts import PromptTemplate
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    logger.warning("langchain not installed -- agent tool-use unavailable.")


CLINICAL_BERT_MODEL = os.getenv("CLINICAL_BERT_MODEL", "medicalai/ClinicalBERT")

_tokenizer = None
_bert_model = None


def _get_clinical_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        logger.info("Loading ClinicalBERT: %s", CLINICAL_BERT_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
        _bert_model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL)
        _bert_model.eval()
    return _tokenizer, _bert_model


def encode_texts(texts: list[str]) -> "np.ndarray":
    if not _BERT_AVAILABLE:
        raise RuntimeError("transformers not installed.")

    tokenizer, model = _get_clinical_bert()
    embeddings = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
CHROMA_COLLECTION = "radiology_reports"

_chroma_client = None
_chroma_collection = None


def _get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready at %s — collection '%s' has %d documents.",
            CHROMA_DIR, CHROMA_COLLECTION, _chroma_collection.count(),
        )
    return _chroma_client, _chroma_collection


_faiss_index = None
_indexed_reports: list[str] = []

FAISS_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_store")
FAISS_INDEX_FILE = os.path.join(FAISS_DIR, "reports.index")
FAISS_TEXTS_FILE = os.path.join(FAISS_DIR, "reports.json")


def _load_faiss():
    global _faiss_index, _indexed_reports
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_TEXTS_FILE):
        _faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_TEXTS_FILE) as f:
            _indexed_reports = json.load(f)
        logger.info("FAISS index loaded from disk (%d vectors).", _faiss_index.ntotal)


def _save_faiss():
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(_faiss_index, FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "w") as f:
        json.dump(_indexed_reports, f)


if _FAISS_AVAILABLE and not _CHROMA_AVAILABLE:
    try:
        _load_faiss()
    except Exception as e:
        logger.warning("Could not load FAISS index: %s", e)


def build_index(reports: list[str], metadatas: list[dict] = None):
    global _faiss_index, _indexed_reports

    if not _BERT_AVAILABLE:
        logger.error("Cannot build index — transformers not installed.")
        return

    metadatas = metadatas or [{} for _ in reports]
    embeddings = encode_texts(reports)

    if _CHROMA_AVAILABLE:
        _, collection = _get_chroma()
        start_id = collection.count()
        ids = [str(start_id + i) for i in range(len(reports))]
        collection.add(
            ids=ids,
            documents=reports,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info("ChromaDB: added %d reports. Total: %d.", len(reports), collection.count())
        return

    if _FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        _faiss_index = index
        _indexed_reports = reports
        _save_faiss()
        logger.info("FAISS index built: %d vectors.", index.ntotal)
        return

    logger.error("No vector store available — install chromadb or faiss-cpu.")


def retrieve_similar_reports(
    query: str,
    k: int = 3,
    filters: dict = None,
) -> list[dict]:
    if not _BERT_AVAILABLE:
        return []

    q_emb = encode_texts([query])

    if _CHROMA_AVAILABLE:
        _, collection = _get_chroma()
        if collection.count() == 0:
            return []

        query_kwargs = dict(
            query_embeddings=q_emb.tolist(),
            n_results=min(k, collection.count()),
            include=["documents", "distances", "metadatas"],
        )
        if filters:
            query_kwargs["where"] = filters

        results = collection.query(**query_kwargs)

        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            score = round(1 - dist / 2, 4)
            output.append({"text": doc, "score": score, "metadata": meta})
        return output

    if _FAISS_AVAILABLE and _faiss_index is not None:
        _, indices = _faiss_index.search(q_emb, k)
        return [
            {"text": _indexed_reports[i], "score": None, "metadata": {}}
            for i in indices[0] if i < len(_indexed_reports)
        ]

    return []


def delete_collection():
    if _CHROMA_AVAILABLE:
        client, _ = _get_chroma()
        client.delete_collection(CHROMA_COLLECTION)
        global _chroma_collection
        _chroma_collection = None
        logger.info("ChromaDB collection '%s' deleted.", CHROMA_COLLECTION)


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


class _DemoLLM:
    _FINDINGS = {
        "cardiomegaly":    "cardiomegaly",
        "cardiac":         "cardiomegaly",
        "heart":           "cardiomegaly",
        "pleural":         "pleural effusion",
        "effusion":        "pleural effusion",
        "pneumothorax":    "pneumothorax",
        "consolidation":   "consolidation",
        "atelectasis":     "atelectasis",
        "collapse":        "atelectasis",
    }

    def _detect_finding(self, text: str) -> str:
        lower = text.lower()
        for kw, finding in self._FINDINGS.items():
            if kw in lower:
                return finding
        return "consolidation"

    def _predict(self, prompt: str) -> str:
        question = ""
        for line in reversed(prompt.splitlines()):
            line = line.strip()
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
                break

        finding = self._detect_finding(question)
        search_q = f"{finding} radiology report findings"

        if "Observation:" not in prompt:
            return (
                f"Thought: I should search the knowledge base for cases related to {finding}.\n"
                f"Action: search_similar_cases\n"
                f"Action Input: {search_q}"
            )

        if prompt.count("Observation:") == 1:
            return (
                f"Thought: Now I should get the clinical explanation for {finding}.\n"
                f"Action: explain_finding\n"
                f"Action Input: {finding}"
            )

        return (
            f"Thought: I now have enough information to answer the question.\n"
            f"Final Answer: {finding.capitalize()} refers to a significant radiological "
            f"finding on chest X-ray. Based on similar cases retrieved from the knowledge "
            f"base and the clinical definition, this finding requires prompt clinical "
            f"correlation. Please consult a qualified radiologist for patient-specific advice."
        )

    def __call__(self, prompt: str, **kwargs) -> str:
        return self._predict(prompt)

    def invoke(self, prompt, **kwargs) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return self._predict(text)

    @property
    def _llm_type(self) -> str:
        return "demo"

    def predict(self, text: str, **kwargs) -> str:
        return self._predict(text)


def _get_llm():
    if VLLM_BASE_URL:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Using vLLM backend at %s", VLLM_BASE_URL)
            return ChatOpenAI(
                base_url=f"{VLLM_BASE_URL}/v1",
                api_key="EMPTY",
                model=VLLM_MODEL,
                temperature=0.2,
            )
        except ImportError:
            logger.warning("langchain_openai not installed.")

    logger.info("Using DemoLLM (rule-based). Set VLLM_BASE_URL for a real LLM.")
    return _DemoLLM()


_REACT_PROMPT = PromptTemplate.from_template(
    "You are a clinical AI assistant specialising in chest X-ray radiology.\n"
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format EXACTLY:\n"
    "Question: the input question you must answer\n"
    "Thought: your reasoning step\n"
    "Action: one of [{tool_names}]\n"
    "Action Input: the input to the tool\n"
    "Observation: the result of the action\n"
    "... (Thought/Action/Action Input/Observation can repeat up to 5 times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original question\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
) if _LANGCHAIN_AVAILABLE else None


def _build_agent_tools(report_context: str):
    def search_similar_cases(query: str) -> str:
        results = retrieve_similar_reports(query, k=3)
        if not results:
            return "No similar cases found in the knowledge base."
        lines = []
        for r in results:
            score_str = f" (similarity: {r['score']})" if r["score"] else ""
            meta_str = f" | {r['metadata']}" if r["metadata"] else ""
            lines.append(f"{r['text']}{score_str}{meta_str}")
        return "\n---\n".join(lines)

    def explain_finding(finding: str) -> str:
        explanations = {
            "cardiomegaly": "Enlargement of the cardiac silhouette (>50% of thoracic width on PA view). Associated with heart failure, pericardial effusion, dilated cardiomyopathy.",
            "pleural effusion": "Fluid in the pleural space. Appears as blunting of costophrenic angles. Causes include CHF, pneumonia, malignancy.",
            "pneumothorax": "Air in the pleural space. Visible as a thin pleural line with absent lung markings peripherally.",
            "consolidation": "Airspace opacification indicating fluid/cells replacing air in alveoli. Caused by pneumonia, pulmonary edema, hemorrhage.",
            "atelectasis": "Lung collapse, partial or complete. Shows as increased opacity with volume loss.",
        }
        key = finding.lower().strip()
        for k, v in explanations.items():
            if k in key:
                return v
        return f"No pre-loaded explanation for '{finding}'. Consult radiological literature."

    def summarise_report(text: str) -> str:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        return ". ".join(sentences[:2]) + "." if sentences else text

    return [
        Tool(name="search_similar_cases", func=search_similar_cases,
             description="Search the knowledge base for radiological reports similar to a query. Input: a clinical description or finding."),
        Tool(name="explain_finding", func=explain_finding,
             description="Get a clinical explanation of a radiology finding. Input: the finding name (e.g. 'cardiomegaly')."),
        Tool(name="summarise_report", func=summarise_report,
             description="Summarise a radiology report to its key findings. Input: full report text."),
    ]


def query_agent(question: str, report_context: str = "") -> str:
    return query_agent_with_trace(question, report_context)["answer"]


def query_agent_with_trace(question: str, report_context: str = "") -> dict:
    if not _LANGCHAIN_AVAILABLE:
        results = retrieve_similar_reports(question)
        if results:
            texts = [r["text"] for r in results]
            answer = "Similar cases found:\n" + "\n---\n".join(texts)
        else:
            answer = "LangChain not installed. Install langchain langchain-community to enable the agent."
        return {"answer": answer, "langchain_active": False, "steps": []}

    llm = _get_llm()
    tools = _build_agent_tools(report_context)
    tool_map = {t.name: t for t in tools}
    full_question = question
    if report_context:
        full_question = f"Given this radiology report: '{report_context}'\n\nQuestion: {question}"

    if isinstance(llm, _DemoLLM):
        return _run_demo_agent(llm, tools, tool_map, full_question)

    try:
        agent = create_react_agent(llm, tools, _REACT_PROMPT)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
        )
        result = executor.invoke({"input": full_question})

        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append({
                "thought":    getattr(action, "log", "").strip(),
                "tool":       action.tool,
                "tool_input": str(action.tool_input),
                "observation": str(observation),
            })

        return {
            "answer": result.get("output", "Agent returned no output."),
            "langchain_active": True,
            "steps": steps,
        }
    except Exception as exc:
        logger.exception("Agent execution failed: %s", exc)
        return {"answer": f"Agent error: {exc}", "langchain_active": False, "steps": []}


def _run_demo_agent(llm: "_DemoLLM", tools, tool_map: dict, question: str) -> dict:
    steps = []
    scratchpad = ""
    max_iter = 5

    for _ in range(max_iter):
        prompt = (
            "You are a clinical AI assistant.\n\n"
            f"Question: {question}\n"
            f"{scratchpad}"
        )
        raw = llm._predict(prompt)

        if "Final Answer:" in raw:
            answer = raw.split("Final Answer:")[-1].strip()
            return {"answer": answer, "langchain_active": True, "steps": steps}

        thought = ""
        action = ""
        action_input = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                action_input = line[len("Action Input:"):].strip()

        if not action or action not in tool_map:
            break

        try:
            observation = tool_map[action].func(action_input)
        except Exception as exc:
            observation = f"Tool error: {exc}"

        steps.append({
            "thought":     thought,
            "tool":        action,
            "tool_input":  action_input,
            "observation": str(observation),
        })

        scratchpad += (
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {observation}\n"
        )

    return {
        "answer": "Agent completed. See reasoning trace above for details.",
        "langchain_active": True,
        "steps": steps,
    }
