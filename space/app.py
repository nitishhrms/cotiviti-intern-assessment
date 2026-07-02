"""
Chest X-Ray Report Generation + Clinical RAG Agent
Hugging Face Space app.

Tab 1  Report Generation : DenseNet-121 encoder -> Transformer decoder
Tab 2  Clinical Agent    : RAG (ClinicalBERT embeddings, cosine retrieval) + ReAct trace
Tab 3  Knowledge Base    : semantic search over the report corpus + add reports
Tab 4  About

Educational / research demo only. Not FDA-cleared for clinical use.
"""

import os
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ============================================================================
# Report-generation model (DenseNet-121 encoder + Transformer decoder)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(1024, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, images):
        x = self.features(images)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3, dropout=0.1, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, captions, image_features, tgt_mask=None, tgt_key_padding_mask=None):
        x = self.embedding(captions)
        x = self.pos_encoding(x)
        memory = image_features.unsqueeze(1)
        out = self.transformer_decoder(
            tgt=x, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)


class RadiologyReportModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3, dropout=0.1, max_len=100):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TextDecoder(vocab_size, embed_dim, num_heads, num_layers, dropout, max_len)

    def generate(self, image, word2idx, idx2word, max_len=100, device="cpu"):
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
            image_features = self.encoder(image)
            tokens = [word2idx["<start>"]]
            for _ in range(max_len):
                caption = torch.tensor([tokens], dtype=torch.long).to(device)
                logits = self.decoder(caption, image_features)
                next_token = logits[0, -1, :].argmax().item()
                tokens.append(next_token)
                if idx2word[next_token] == "<end>":
                    break
        words = [idx2word[t] for t in tokens if idx2word[t] not in ["<start>", "<end>", "<pad>"]]
        return " ".join(words)


MODEL_DIR = "xray_caption_models"
VOCAB_PATH = "vocab.json"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)
with open(VOCAB_PATH) as f:
    vocab_data = json.load(f)
word2idx = vocab_data["word2idx"]
idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

model = RadiologyReportModel(
    vocab_size=len(word2idx), embed_dim=CONFIG["embed_dim"], num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"], dropout=CONFIG["dropout"], max_len=CONFIG["max_len"],
).to(device)

MODEL_OK = False
if os.path.exists(BEST_MODEL_PATH):
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    MODEL_OK = True
    print("Report model loaded.")
else:
    print("WARNING: model weights not found.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_report(image):
    if image is None:
        return "Please upload a chest X-ray image."
    if not MODEL_OK:
        return "Model weights are not available on this Space."
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image).to(device)
    report = model.generate(img_tensor, word2idx, idx2word, max_len=CONFIG["max_len"], device=device)
    return report if report.strip() else "(model produced an empty report; try another image)"


# ============================================================================
# Clinical RAG Agent  (ClinicalBERT embeddings -> cosine retrieval -> ReAct)
# ============================================================================
CORPUS = [
    {"text": "Cardiomegaly with mild pulmonary vascular congestion. No focal consolidation.", "label": "cardiomegaly"},
    {"text": "Enlarged cardiac silhouette consistent with cardiomegaly. Lungs are clear.", "label": "cardiomegaly"},
    {"text": "Small bilateral pleural effusions with associated basilar atelectasis.", "label": "pleural effusion"},
    {"text": "Moderate right pleural effusion. No pneumothorax.", "label": "pleural effusion"},
    {"text": "Right lower lobe consolidation consistent with pneumonia.", "label": "pneumonia"},
    {"text": "Patchy left basilar airspace opacity, likely pneumonia in the correct clinical setting.", "label": "pneumonia"},
    {"text": "Left apical pneumothorax. Recommend clinical correlation and possible chest tube.", "label": "pneumothorax"},
    {"text": "Small right-sided pneumothorax without tension.", "label": "pneumothorax"},
    {"text": "Bibasilar atelectasis. No acute cardiopulmonary process.", "label": "atelectasis"},
    {"text": "Pulmonary edema with cephalization and interstitial markings.", "label": "edema"},
    {"text": "Findings consistent with congestive heart failure and pulmonary edema.", "label": "edema"},
    {"text": "No acute cardiopulmonary abnormality. Clear lungs and normal heart size.", "label": "no finding"},
    {"text": "Endotracheal tube and central venous catheter in standard position.", "label": "support devices"},
    {"text": "Hyperinflated lungs with flattened diaphragms suggesting emphysema.", "label": "lung opacity"},
    {"text": "Healed rib fractures on the right. No acute osseous injury.", "label": "fracture"},
    {"text": "Diffuse bilateral airspace opacities consistent with pulmonary edema or ARDS.", "label": "edema"},
]

FINDINGS_KB = {
    "cardiomegaly": "Cardiomegaly is enlargement of the cardiac silhouette, typically a cardiothoracic ratio "
                    "greater than 0.5 on a PA chest radiograph. It can indicate heart failure, cardiomyopathy, "
                    "or pericardial effusion, and warrants clinical correlation.",
    "pleural effusion": "A pleural effusion is a collection of fluid in the pleural space, often seen as blunting "
                        "of the costophrenic angle. Causes include heart failure, infection, and malignancy.",
    "pneumothorax": "A pneumothorax is air in the pleural space that can cause partial or complete lung collapse. "
                    "A large or tension pneumothorax is an emergency requiring decompression.",
    "pneumonia": "Pneumonia appears as focal or patchy airspace consolidation, frequently with air bronchograms, "
                 "and should be interpreted together with the clinical picture.",
    "atelectasis": "Atelectasis is collapse or incomplete expansion of lung tissue, commonly seen at the lung "
                   "bases as linear or band-like opacity with volume loss.",
    "edema": "Pulmonary edema is fluid accumulation in the lungs, shown by vascular cephalization, interstitial "
             "markings, and in severe cases airspace opacities; it is often related to heart failure.",
    "consolidation": "Consolidation is airspace filling that replaces air with fluid, pus, or cells, producing a "
                     "dense opacity that may contain air bronchograms.",
    "fracture": "A rib or bony fracture appears as a cortical break or lucent line; acute fractures should be "
                "correlated with the site of pain and mechanism of injury.",
}

SYNONYMS = {
    "enlarged heart": "cardiomegaly", "heart enlargement": "cardiomegaly",
    "fluid around the lung": "pleural effusion", "effusion": "pleural effusion",
    "collapsed lung": "pneumothorax", "lung infection": "pneumonia",
    "fluid in the lungs": "edema", "collapse": "atelectasis",
}

_STATE = {"embed": None, "backend": None, "corpus_emb": None}


def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


def _make_embedder():
    """Prefer ClinicalBERT embeddings; fall back to TF-IDF for reliability."""
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        mdl = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
        mdl.eval()

        def embed(texts):
            enc = tok(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            return _normalize((summed / counts).cpu().numpy())

        return embed, "ClinicalBERT (medicalai/ClinicalBERT)"
    except Exception as exc:  # pragma: no cover
        print("ClinicalBERT unavailable, using TF-IDF fallback:", exc)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer()
        vec.fit([c["text"] for c in CORPUS] + list(FINDINGS_KB.values()))

        def embed(texts):
            return _normalize(vec.transform(list(texts)).toarray())

        return embed, "TF-IDF (fallback retriever)"


def ensure_index():
    if _STATE["embed"] is None:
        embed, backend = _make_embedder()
        _STATE["embed"] = embed
        _STATE["backend"] = backend
        _STATE["corpus_emb"] = embed([c["text"] for c in CORPUS])


def retrieve(query, k=3):
    ensure_index()
    q = _STATE["embed"]([query])[0]
    sims = _STATE["corpus_emb"] @ q
    order = np.argsort(sims)[::-1][:k]
    return [(CORPUS[i]["text"], CORPUS[i]["label"], float(sims[i])) for i in order]


def detect_finding(text):
    t = text.lower()
    for phrase, key in SYNONYMS.items():
        if phrase in t:
            return key
    for key in FINDINGS_KB:
        if key in t:
            return key
    return None


def agent_answer(question, k=3):
    if not question or not question.strip():
        return "Please enter a clinical question.", "", ""
    ensure_index()
    trace = []
    trace.append("Question: " + question.strip())
    trace.append("")
    trace.append("Step 1")
    trace.append("  Thought: Search the report knowledge base for similar cases.")
    trace.append(f"  Action:  search_similar_cases(\"{question.strip()}\")")
    hits = retrieve(question, k)
    for txt, lbl, s in hits:
        trace.append(f"  Obs:     [{lbl}] {txt}  (similarity={s:.2f})")

    finding = detect_finding(question) or (hits[0][1] if hits else None)
    explanation = FINDINGS_KB.get(finding) if finding else None
    if explanation:
        trace.append("")
        trace.append("Step 2")
        trace.append(f"  Thought: Explain the key finding '{finding}'.")
        trace.append(f"  Action:  explain_finding(\"{finding}\")")
        trace.append(f"  Obs:     {explanation}")

    parts = []
    if explanation:
        parts.append(explanation)
    if hits:
        examples = "; ".join(h[0] for h in hits[:2])
        parts.append("Grounded in similar documented cases: " + examples)
    answer = " ".join(parts) if parts else "No relevant clinical information was found for that query."

    trace.append("")
    trace.append("Final Answer:")
    trace.append("  " + answer)
    backend = f"Retriever: **{_STATE['backend']}**  ·  {len(CORPUS)} indexed reports"
    return answer, "\n".join(trace), backend


def kb_search(query, k=3):
    if not query or not query.strip():
        return [], ""
    hits = retrieve(query, k)
    rows = [[f"{s:.2f}", lbl, txt] for txt, lbl, s in hits]
    ensure_index()
    return rows, f"Retriever: **{_STATE['backend']}**  ·  {len(CORPUS)} indexed reports"


def kb_add(report, label):
    if not report or not report.strip():
        return "Enter a report to add.", _kb_count()
    ensure_index()
    CORPUS.append({"text": report.strip(), "label": (label or "user").strip()})
    new_emb = _STATE["embed"]([report.strip()])
    _STATE["corpus_emb"] = np.vstack([_STATE["corpus_emb"], new_emb])
    return f"Added. The knowledge base now has {len(CORPUS)} reports.", _kb_count()


def _kb_count():
    return f"**{len(CORPUS)}** reports currently indexed."


# ============================================================================
# Gradio UI
# ============================================================================
DISCLAIMER = ("_Educational and research demo only. Not FDA-cleared and must not be used "
              "for clinical diagnosis or patient care._")

with gr.Blocks(title="Chest X-Ray Report Generation + Clinical Agent",
               theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")) as demo:
    gr.Markdown(
        "# 🩻 Chest X-Ray Report Generation + Clinical RAG Agent\n"
        "DenseNet-121 + Transformer report generation, with a Retrieval-Augmented clinical "
        "question-answering agent that shows its reasoning trace.\n\n" + DISCLAIMER
    )

    with gr.Tabs():
        # ---- Tab 1: Report Generation ----
        with gr.Tab("📄 Report Generation"):
            gr.Markdown("Upload a chest X-ray. The vision encoder and Transformer decoder generate a draft report.")
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Chest X-Ray (JPEG / PNG)")
                    gen_btn = gr.Button("Generate report", variant="primary")
                with gr.Column():
                    report_out = gr.Textbox(label="Generated Radiology Report", lines=8)
            gen_btn.click(predict_report, inputs=img_in, outputs=report_out)

        # ---- Tab 2: Clinical Agent ----
        with gr.Tab("🤖 Clinical Agent"):
            gr.Markdown("Ask a clinical question. The agent retrieves similar cases, explains the key finding, "
                        "and answers, showing a ReAct-style reasoning trace.")
            q_in = gr.Textbox(label="Clinical question", value="What does cardiomegaly indicate?")
            k_in = gr.Slider(1, 5, value=3, step=1, label="Number of cases to retrieve (top-k)")
            ask_btn = gr.Button("Ask the agent", variant="primary")
            ans_out = gr.Textbox(label="Answer", lines=4)
            trace_out = gr.Textbox(label="Reasoning trace (ReAct)", lines=14)
            backend_out = gr.Markdown()
            ask_btn.click(agent_answer, inputs=[q_in, k_in], outputs=[ans_out, trace_out, backend_out])
            gr.Examples(
                [["What does cardiomegaly indicate?"],
                 ["Explain bilateral pleural effusions"],
                 ["What is a pneumothorax and is it dangerous?"],
                 ["What causes pulmonary edema?"]],
                inputs=[q_in],
            )

        # ---- Tab 3: Knowledge Base ----
        with gr.Tab("🔎 Knowledge Base"):
            gr.Markdown("Semantic search over the report corpus, and add your own reports (RAG index management).")
            with gr.Row():
                kq_in = gr.Textbox(label="Search query", value="fluid around the lung")
                kk_in = gr.Slider(1, 5, value=3, step=1, label="Top-k")
            ksearch_btn = gr.Button("Search", variant="primary")
            kb_table = gr.Dataframe(headers=["Score", "Finding", "Report"], label="Retrieved reports", wrap=True)
            kb_backend = gr.Markdown()
            ksearch_btn.click(kb_search, inputs=[kq_in, kk_in], outputs=[kb_table, kb_backend])

            gr.Markdown("### Add a report to the knowledge base")
            with gr.Row():
                add_text = gr.Textbox(label="Report text")
                add_label = gr.Textbox(label="Finding label (optional)")
            add_btn = gr.Button("Add report")
            add_status = gr.Markdown()
            kb_count = gr.Markdown(_kb_count())
            add_btn.click(kb_add, inputs=[add_text, add_label], outputs=[add_status, kb_count])

        # ---- Tab 4: About ----
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                "### How it works\n"
                "- **Report generation:** a DenseNet-121 encoder turns the X-ray into a feature vector, "
                "and a 3-layer Transformer decoder generates the report (trained on the Indiana University "
                "Chest X-ray Collection).\n"
                "- **Clinical agent (RAG):** the question is embedded with **ClinicalBERT**, similar reports "
                "are retrieved by cosine similarity, the key finding is explained from a curated knowledge "
                "base, and the answer is synthesized, grounded in the retrieved cases. A **ReAct** trace "
                "shows every step.\n"
                "- **Knowledge base:** semantic search and index management over the report corpus.\n\n"
                "Full source, MLOps backend (FastAPI, Celery, Prometheus, Docker) and documentation: "
                "[GitHub](https://github.com/nitishhrms/cotiviti-intern-assessment).\n\n" + DISCLAIMER
            )

if __name__ == "__main__":
    demo.launch()
