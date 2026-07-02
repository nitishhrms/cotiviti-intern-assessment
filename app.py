import io
import json
import math
import os
import time

import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import gradio as gr

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT  = int(os.getenv("API_TIMEOUT_SECONDS", "30"))


def _api(method: str, path: str, **kwargs):
    url = f"{API_BASE_URL}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=API_TIMEOUT, **kwargs)
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                return resp.json(), None
            return resp.text, None
        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach API at {url}\nStart the backend: uvicorn backend.main:app --reload"
    except Exception as exc:
        return None, str(exc)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features   = densenet.features
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(1024, embed_dim)
        self.relu       = nn.ReLU()

    def forward(self, images):
        x = self.features(images)
        x = self.relu(x)
        x = self.pool(x)
        return self.projection(x.view(x.size(0), -1))


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=3, dropout=0.1, max_len=100):
        super().__init__()
        self.embedding         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding      = PositionalEncoding(embed_dim, max_len, dropout)
        decoder_layer          = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=512, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out            = nn.Linear(embed_dim, vocab_size)

    def forward(self, captions, image_features, tgt_mask=None, tgt_key_padding_mask=None):
        x   = self.pos_encoding(self.embedding(captions))
        out = self.transformer_decoder(
            tgt=x, memory=image_features.unsqueeze(1),
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)


class RadiologyReportModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=3, dropout=0.1, max_len=100):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TextDecoder(vocab_size, embed_dim, num_heads,
                                   num_layers, dropout, max_len)

    def forward(self, images, captions):
        seq_len              = captions.size(1)
        tgt_mask             = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        tgt_key_padding_mask = captions == 0
        return self.decoder(captions, self.encoder(images), tgt_mask, tgt_key_padding_mask)

    def generate(self, image, word2idx, idx2word, max_len=100, device="cpu"):
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image_features = self.encoder(image.to(device))
            tokens = [word2idx["<start>"]]
            for _ in range(max_len):
                caption    = torch.tensor([tokens], dtype=torch.long).to(device)
                logits     = self.decoder(caption, image_features)
                next_token = logits[0, -1, :].argmax().item()
                tokens.append(next_token)
                if idx2word[next_token] == "<end>":
                    break
        return " ".join(
            idx2word[t] for t in tokens
            if idx2word[t] not in ("<start>", "<end>", "<pad>")
        )


MODEL_DIR       = "xray_caption_models"
VOCAB_PATH      = "vocab.json"
CONFIG_PATH     = os.path.join(MODEL_DIR, "config.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_loaded = False
word2idx = {}
idx2word = {}
CONFIG   = {}
model    = None

try:
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
    with open(VOCAB_PATH) as f:
        vocab_data = json.load(f)
    word2idx = vocab_data["word2idx"]
    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

    model = RadiologyReportModel(
        vocab_size=len(word2idx),
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_len"],
    ).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model_loaded = True
        print(f"Model loaded on {device}")
    else:
        print(f"WARNING: model weights not found at {BEST_MODEL_PATH}")
except Exception as exc:
    print(f"WARNING: model load failed -- {exc}")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    import pydicom
    _PYDICOM = True
except ImportError:
    _PYDICOM = False

_WINDOW_PRESETS = {
    "Auto (from DICOM)": "auto",
    "Lung":              "lung",
    "Mediastinum":       "mediastinum",
    "Bone":              "bone",
}
_SEX_MAP = {"Male": 0, "Female": 1, "Not Specified": 2}


def _load_dicom_to_pil(dicom_bytes: bytes, window_preset: str):
    from src.data.dicom_loader import load_dicom, apply_window, auto_window, WINDOW_PRESETS
    result      = load_dicom(dicom_bytes)
    pixel_array = result["pixel_array"]
    ds          = result["ds"]
    patient_info = result["patient_info"]
    params      = auto_window(ds) if window_preset == "auto" \
                  else WINDOW_PRESETS.get(window_preset, WINDOW_PRESETS["mediastinum"])
    windowed    = apply_window(pixel_array, params["center"], params["width"])
    if windowed.ndim == 3:
        windowed = windowed[0]
    pil_image = Image.fromarray(windowed, mode="L").convert("RGB")
    raw       = patient_info.get("raw", {})
    meta_str  = (
        f"Age: {raw.get('PatientAge') or 'N/A'}  |  "
        f"Sex: {raw.get('PatientSex') or 'N/A'}  |  "
        f"Modality: {raw.get('Modality') or 'N/A'}  |  "
        f"View: {result['metadata'].get('ViewPosition', 'N/A')}"
    )
    return pil_image, meta_str


def predict_report(image, patient_age, patient_sex, window_preset_label, dicom_file):
    t0       = time.perf_counter()
    meta_str = "No metadata available."

    if dicom_file is not None:
        if not _PYDICOM:
            return "pydicom not installed. Run: pip install pydicom", "", ""
        try:
            dicom_bytes = open(dicom_file, "rb").read() \
                if isinstance(dicom_file, str) else dicom_file.read()
            preset_key  = _WINDOW_PRESETS.get(window_preset_label, "mediastinum")
            image, meta_str = _load_dicom_to_pil(dicom_bytes, preset_key)
        except Exception as exc:
            return f"DICOM loading failed: {exc}", "", ""

    if image is None:
        return "Please upload an X-ray image or DICOM file.", "", ""

    if not _model_loaded:
        return "Model weights not found -- check xray_caption_models/best_model.pth", meta_str, ""

    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    img_tensor = _transform(pil_image).to(device)
    report     = model.generate(img_tensor, word2idx, idx2word,
                                max_len=CONFIG.get("max_len", 100), device=device)
    elapsed    = round((time.perf_counter() - t0) * 1000)
    perf_str   = f"Processing time: {elapsed} ms  |  Device: {device}  |  Model v1.0.0"
    return report, meta_str, perf_str


def agent_query_gradio(question: str, report_context: str):
    if not question.strip():
        return "No question entered.", "", ""

    try:
        from backend.rag_agent import query_agent_with_trace
        result = query_agent_with_trace(question.strip(), report_context.strip())
    except Exception as exc:
        return f"Import error: {exc}", "", ""

    if result["langchain_active"]:
        status = "LangChain ACTIVE -- full ReAct agent loop running"
    else:
        status = "LangChain NOT active -- showing raw retrieval results (install langchain + LLM)"

    if not result["steps"]:
        trace = "(No tool calls made -- agent answered directly from its training knowledge.)"
    else:
        lines = []
        for i, step in enumerate(result["steps"], 1):
            lines.append(f"+= Step {i} {'=' * 50}")
            if step["thought"]:
                thought_lines = [
                    l.strip() for l in step["thought"].splitlines()
                    if l.strip().startswith("Thought:")
                ]
                thought = thought_lines[-1].removeprefix("Thought:").strip() \
                    if thought_lines else step["thought"].splitlines()[-1].strip()
                lines.append(f"|  Thought     : {thought}")
            lines.append(f"|  Tool        : {step['tool']}")
            lines.append(f"|  Input       : {step['tool_input']}")
            obs = step["observation"]
            if len(obs) > 500:
                obs = obs[:500] + " ...[truncated]"
            lines.append(f"|  Observation : {obs}")
            lines.append(f"+{'=' * 56}")
            lines.append("")
        trace = "\n".join(lines)

    return status, trace, result["answer"]


def kb_add_reports(reports_text: str, metadata_json: str):
    reports = [r.strip() for r in reports_text.strip().split("\n\n") if r.strip()]
    if not reports:
        return "No reports found. Separate multiple reports with a blank line."

    metadatas = []
    if metadata_json.strip():
        try:
            metadatas = json.loads(metadata_json)
            if not isinstance(metadatas, list):
                return "Metadata must be a JSON array: [{...}, {...}]"
        except json.JSONDecodeError as exc:
            return f"Invalid JSON metadata: {exc}"
    else:
        metadatas = [{} for _ in reports]

    if metadatas and len(metadatas) != len(reports):
        return (f"Mismatch: {len(reports)} reports but {len(metadatas)} metadata entries. "
                "Either omit metadata or provide one entry per report.")
    try:
        from backend.rag_agent import build_index
        build_index(reports, metadatas)
        return f"Successfully added {len(reports)} report(s) to the knowledge base."
    except Exception as exc:
        return f"Error: {exc}"


def kb_search(query: str, k: int, filters_json: str):
    if not query.strip():
        return "Please enter a search query.", []

    filters = None
    if filters_json.strip():
        try:
            filters = json.loads(filters_json)
        except json.JSONDecodeError as exc:
            return f"Invalid filters JSON: {exc}", []

    try:
        from backend.rag_agent import retrieve_similar_reports
        results = retrieve_similar_reports(query.strip(), k=int(k), filters=filters)
    except Exception as exc:
        return f"Error: {exc}", []

    if not results:
        return "No results found. Try adding reports first (Knowledge Base -> Add Reports).", []

    rows = []
    for i, r in enumerate(results, 1):
        score = f"{r['score']:.4f}" if r["score"] is not None else "N/A"
        meta  = json.dumps(r["metadata"]) if r["metadata"] else "{}"
        text  = r["text"][:200] + ("..." if len(r["text"]) > 200 else "")
        rows.append([i, score, meta, text])

    status = f"Found {len(results)} result(s) for query: '{query}'"
    return status, rows


def kb_stats():
    try:
        from backend.rag_agent import (
            _CHROMA_AVAILABLE, _FAISS_AVAILABLE,
            _faiss_index, _indexed_reports,
        )
        lines = []
        if _CHROMA_AVAILABLE:
            from backend.rag_agent import _get_chroma, CHROMA_DIR, CHROMA_COLLECTION
            _, col = _get_chroma()
            lines.append(f"Vector Store : ChromaDB (persistent)")
            lines.append(f"Location     : {os.path.abspath(CHROMA_DIR)}")
            lines.append(f"Collection   : {CHROMA_COLLECTION}")
            lines.append(f"Documents    : {col.count()}")
        elif _FAISS_AVAILABLE:
            n = _faiss_index.ntotal if _faiss_index else 0
            lines.append(f"Vector Store : FAISS (in-memory + disk)")
            lines.append(f"Documents    : {n}")
            lines.append(f"Texts cached : {len(_indexed_reports)}")
        else:
            lines.append("No vector store available.")
            lines.append("Install: pip install chromadb   OR   pip install faiss-cpu")

        try:
            from backend.rag_agent import _BERT_AVAILABLE, CLINICAL_BERT_MODEL
            lines.append(f"Encoder      : {'ClinicalBERT (' + CLINICAL_BERT_MODEL + ')' if _BERT_AVAILABLE else 'NOT available (install transformers)'}")
        except Exception:
            pass

        try:
            from backend.rag_agent import _LANGCHAIN_AVAILABLE
            lines.append(f"LangChain    : {'installed' if _LANGCHAIN_AVAILABLE else 'NOT installed'}")
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading stats: {exc}"


def kb_delete():
    try:
        from backend.rag_agent import delete_collection, _CHROMA_AVAILABLE
        if not _CHROMA_AVAILABLE:
            return "ChromaDB not installed -- nothing to delete."
        delete_collection()
        return "Collection deleted. The knowledge base is now empty."
    except Exception as exc:
        return f"Error: {exc}"


def telemetry_health():
    data, err = _api("get", "/health")
    if err:
        return f"API OFFLINE\n\n{err}"
    lines = [
        f"Status        : {data.get('status', '?').upper()}",
        f"Model loaded  : {data.get('model_loaded', '?')}",
        f"Version       : {data.get('version', '?')}",
        f"Uptime        : {data.get('uptime_seconds', '?')} seconds",
        f"API endpoint  : {API_BASE_URL}/health",
    ]
    return "\n".join(lines)


def telemetry_model_info():
    data, err = _api("get", "/metadata")
    if err:
        return f"Could not fetch metadata:\n{err}"
    lines = [
        f"Model Version      : {data.get('model_version', '?')}",
        f"Vocabulary Size    : {data.get('vocab_size', '?'):,}",
        f"Embedding Dim      : {data.get('embed_dim', '?')}",
        f"Attention Heads    : {data.get('num_heads', '?')}",
        f"Decoder Layers     : {data.get('num_layers', '?')}",
        f"Max Output Length  : {data.get('max_len', '?')} tokens",
        f"Training Dataset   : {data.get('training_dataset', '?')}",
    ]
    return "\n".join(lines)


def telemetry_prometheus():
    data, err = _api("get", "/metrics")
    if err:
        return f"Prometheus metrics unavailable:\n{err}"
    if isinstance(data, str):
        metric_lines = [l for l in data.splitlines()
                        if l and not l.startswith("#")]
        header = "# Live Prometheus Counters\n"
        return header + "\n".join(metric_lines[:60])
    return str(data)


def telemetry_otel_status():
    enabled  = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    svc_name = os.getenv("OTEL_SERVICE_NAME", "xray-report-api")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    try:
        import opentelemetry
        otel_pkg = f"installed (v{opentelemetry.__version__})"
    except ImportError:
        otel_pkg = "NOT installed  (pip install opentelemetry-sdk opentelemetry-exporter-otlp)"

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        instr_pkg = "installed"
    except ImportError:
        instr_pkg = "NOT installed  (pip install opentelemetry-instrumentation-fastapi)"

    status_icon = "ENABLED" if enabled else "DISABLED"
    lines = [
        f"OpenTelemetry        : {status_icon}",
        f"Service name         : {svc_name}",
        f"OTLP exporter URL    : {endpoint}",
        f"opentelemetry-sdk    : {otel_pkg}",
        f"FastAPI instrumentor : {instr_pkg}",
        "",
        "To enable: set env var   OTEL_ENABLED=true",
        "To point at Jaeger/Tempo: OTEL_EXPORTER_OTLP_ENDPOINT=http://<host>:4317",
    ]
    return "\n".join(lines)


def telemetry_rag_stats():
    data, err = _api("get", "/agent/stats")
    if err:
        return kb_stats()
    lines = [
        f"Store type   : {data.get('store', '?')}",
        f"Documents    : {data.get('num_reports', '?')}",
    ]
    if "collection" in data:
        lines.append(f"Collection   : {data['collection']}")
    return "\n".join(lines)


def telemetry_refresh():
    return (
        telemetry_health(),
        telemetry_model_info(),
        telemetry_prometheus(),
        telemetry_otel_status(),
        telemetry_rag_stats(),
    )


ACCENT  = "#2196F3"
BG_DARK = "#1a1a2e"
CARD_BG = "#16213e"

custom_css = """
.status-ok  { color: #4CAF50; font-weight: bold; }
.status-err { color: #f44336; font-weight: bold; }
.trace-box  { font-family: 'Courier New', monospace; font-size: 12px; }
.tab-header { font-size: 14px; font-weight: 600; }
footer      { display: none !important; }
"""

with gr.Blocks(title="X-Ray Report Generation -- Full Project UI") as demo:

    gr.Markdown(
        "# Chest X-Ray Report Generation System\n"
        "**Full project interface** -- inference → clinical agent → knowledge base → telemetry\n\n"
        f"> API backend: `{API_BASE_URL}` -- start with `uvicorn backend.main:app --reload`"
    )

    with gr.Tabs():

        with gr.Tab("Report Generation"):
            gr.Markdown(
                "Upload a chest X-ray (JPEG/PNG) **or** a DICOM file. "
                "The DenseNet121 encoder + Transformer decoder generates the impression section."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t1_image  = gr.Image(type="pil", label="X-Ray Image (JPEG / PNG)")
                    t1_dicom  = gr.File(
                        label="DICOM File (.dcm) -- takes priority over image above",
                        file_types=[".dcm"],
                    )
                    gr.Markdown("#### Patient Metadata (optional)")
                    t1_age    = gr.Number(label="Age (years)", minimum=0, maximum=120, value=None)
                    t1_sex    = gr.Radio(
                        choices=["Male", "Female", "Not Specified"],
                        value="Not Specified", label="Sex",
                    )
                    t1_window = gr.Dropdown(
                        choices=list(_WINDOW_PRESETS.keys()),
                        value="Mediastinum", label="DICOM Window Preset",
                    )
                    t1_btn    = gr.Button("Generate Report", variant="primary")

                with gr.Column(scale=1):
                    t1_report = gr.Textbox(label="Generated Report", lines=8)
                    t1_meta   = gr.Textbox(label="DICOM Patient Metadata", lines=2)
                    t1_perf   = gr.Textbox(label="Performance", lines=1)

            with gr.Accordion("Model Architecture", open=False):
                gr.Markdown(
                    "| Component | Details |\n|---|---|\n"
                    "| Image encoder | DenseNet-121 pretrained on ImageNet + projection head |\n"
                    "| Text decoder  | Transformer decoder (3 layers, 8 heads, embed_dim=256) |\n"
                    "| Decoding      | Greedy token-by-token generation up to 100 tokens |\n"
                    "| Training data | Indiana University Chest X-ray Collection |\n"
                    "| Input size    | 224 × 224, normalised (ImageNet mean/std) |"
                )

            t1_btn.click(
                fn=predict_report,
                inputs=[t1_image, t1_age, t1_sex, t1_window, t1_dicom],
                outputs=[t1_report, t1_meta, t1_perf],
            )

        with gr.Tab("Clinical Agent"):
            gr.Markdown(
                "## LangChain ReAct Clinical Agent\n"
                "The agent **reasons step-by-step** using three tools:\n\n"
                "| Tool | Does |\n|---|---|\n"
                "| `search_similar_cases` | Semantic search over indexed radiology reports |\n"
                "| `explain_finding` | Returns clinical definition for a named finding |\n"
                "| `summarise_report` | Condenses a long report to 2 key sentences |\n\n"
                "_Requires: `langchain` + an LLM (set `VLLM_BASE_URL` or install `transformers`)._"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t2_q    = gr.Textbox(
                        label="Clinical Question",
                        placeholder="e.g. What are the implications of cardiomegaly in an elderly patient?",
                        lines=3,
                    )
                    t2_ctx  = gr.Textbox(
                        label="Report Context (optional -- paste a radiology report)",
                        placeholder="Paste any radiology report here to give the agent context.",
                        lines=5,
                    )
                    t2_btn  = gr.Button("Run Agent", variant="primary")

                with gr.Column(scale=1):
                    t2_status = gr.Textbox(label="Agent Status", lines=1, interactive=False)
                    t2_trace  = gr.Textbox(
                        label="Reasoning Trace -- Thought -> Tool -> Observation (step by step)",
                        lines=14, interactive=False, elem_classes=["trace-box"],
                    )
                    t2_answer = gr.Textbox(label="Final Answer", lines=5, interactive=False)

            with gr.Accordion("How the ReAct loop works", open=False):
                gr.Markdown(
                    "```\n"
                    "User question\n"
                    "  Step 1 -> Thought: 'I need to search for similar cases'\n"
                    "           Tool:    search_similar_cases('cardiomegaly elderly')\n"
                    "           Obs:     [3 reports from FAISS/ChromaDB returned]\n"
                    "  Step 2 -> Thought: 'Now I should explain the finding'\n"
                    "           Tool:    explain_finding('cardiomegaly')\n"
                    "           Obs:     'Enlargement of cardiac silhouette...'\n"
                    "  Final  -> Agent synthesises both observations -> Final Answer\n"
                    "```"
                )

            t2_btn.click(
                fn=agent_query_gradio,
                inputs=[t2_q, t2_ctx],
                outputs=[t2_status, t2_trace, t2_answer],
            )

        with gr.Tab("Knowledge Base"):
            gr.Markdown(
                "## RAG Knowledge Base\n"
                "Manage the vector store that backs the Clinical Agent's `search_similar_cases` tool.\n"
                "ChromaDB (preferred) persists to disk. FAISS is the fallback."
            )

            with gr.Tabs():

                with gr.Tab("Add Reports"):
                    gr.Markdown(
                        "Paste one or more radiology reports. "
                        "**Separate multiple reports with a blank line.**"
                    )
                    t3a_reports = gr.Textbox(
                        label="Reports (blank line between each)",
                        placeholder=(
                            "Cardiomegaly is noted. The cardiac silhouette is enlarged. "
                            "Bilateral pleural effusions are present.\n\n"
                            "No acute cardiopulmonary abnormality. Lungs are clear bilaterally."
                        ),
                        lines=10,
                    )
                    t3a_meta = gr.Textbox(
                        label='Metadata JSON (optional) -- array matching report count',
                        placeholder='[{"view": "PA", "modality": "CR", "age": 65}, {"view": "PA", "age": 40}]',
                        lines=3,
                    )
                    t3a_btn    = gr.Button("Add to Knowledge Base", variant="primary")
                    t3a_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    t3a_btn.click(fn=kb_add_reports,
                                  inputs=[t3a_reports, t3a_meta],
                                  outputs=t3a_status)

                with gr.Tab("Semantic Search"):
                    gr.Markdown(
                        "Search the knowledge base with natural language. "
                        "Uses ClinicalBERT embeddings + cosine similarity."
                    )
                    with gr.Row():
                        t3b_query   = gr.Textbox(
                            label="Query",
                            placeholder="pleural effusion elderly patient",
                            lines=2, scale=3,
                        )
                        t3b_k       = gr.Slider(label="Top-K results", minimum=1, maximum=10,
                                                value=3, step=1, scale=1)
                    t3b_filters = gr.Textbox(
                        label='ChromaDB Metadata Filters (optional JSON)',
                        placeholder='{"view": "PA"}   or   {"age": {"$gte": 60}}',
                        lines=1,
                    )
                    t3b_btn     = gr.Button("Search", variant="primary")
                    t3b_status  = gr.Textbox(label="Status", lines=1, interactive=False)
                    t3b_results = gr.Dataframe(
                        headers=["Rank", "Score", "Metadata", "Report (preview)"],
                        datatype=["number", "str", "str", "str"],
                        label="Results",
                        wrap=True,
                    )
                    t3b_btn.click(fn=kb_search,
                                  inputs=[t3b_query, t3b_k, t3b_filters],
                                  outputs=[t3b_status, t3b_results])

                with gr.Tab("Stats & Manage"):
                    t3c_stats_out = gr.Textbox(
                        label="Knowledge Base Statistics",
                        lines=10, interactive=False,
                    )
                    with gr.Row():
                        t3c_refresh_btn = gr.Button("Refresh Stats")
                        t3c_delete_btn  = gr.Button("Delete Entire Collection", variant="stop")
                    t3c_delete_out = gr.Textbox(label="Delete Status", lines=2, interactive=False)

                    t3c_refresh_btn.click(fn=kb_stats, outputs=t3c_stats_out)
                    t3c_delete_btn.click(fn=kb_delete, outputs=t3c_delete_out)
                    demo.load(fn=kb_stats, outputs=t3c_stats_out)

        with gr.Tab("System & Telemetry"):
            gr.Markdown(
                "## System Health and Observability\n"
                "Live status from the FastAPI backend, Prometheus counters, and OpenTelemetry config.\n"
                f"> Polling: `{API_BASE_URL}`"
            )

            t4_refresh = gr.Button("Refresh All", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### API Health")
                    t4_health = gr.Textbox(
                        label="GET /health", lines=6, interactive=False,
                    )

                    gr.Markdown("### Model Info")
                    t4_meta = gr.Textbox(
                        label="GET /metadata", lines=8, interactive=False,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### RAG / Knowledge Base")
                    t4_rag = gr.Textbox(
                        label="GET /agent/stats", lines=5, interactive=False,
                    )

                    gr.Markdown("### Prometheus Metrics")
                    t4_prom = gr.Textbox(
                        label="GET /metrics", lines=10, interactive=False,
                        elem_classes=["trace-box"],
                    )

            gr.Markdown("---")
            gr.Markdown("### OpenTelemetry Configuration")
            t4_otel = gr.Textbox(
                label="OTEL Status (reads env vars directly)",
                lines=16, interactive=False,
                elem_classes=["trace-box"],
            )

            with gr.Accordion("What each metric means", open=False):
                gr.Markdown(
                    "| Metric | Meaning |\n|---|---|\n"
                    "| `inference_count_total` | Total number of `/predict` calls since startup |\n"
                    "| `inference_latency_seconds` | Histogram of inference wall-clock time |\n"
                    "| `error_count_total` | Total 500-class errors |\n"
                    "| `uptime_seconds` | Seconds since server started |\n"
                )

            t4_refresh.click(
                fn=telemetry_refresh,
                outputs=[t4_health, t4_meta, t4_prom, t4_otel, t4_rag],
            )
            demo.load(
                fn=telemetry_refresh,
                outputs=[t4_health, t4_meta, t4_prom, t4_otel, t4_rag],
            )

    gr.Markdown(
        "_Trained on the Indiana University Chest X-ray Collection. "
        "For educational purposes only -- not for clinical use._"
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
