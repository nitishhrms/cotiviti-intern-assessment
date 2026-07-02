"""
Render a clean High-Level Design (HLD) architecture diagram as a PNG,
used in both the Word report and the PowerPoint deck.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

PURPLE = "#4B2E83"
PURPLE_D = "#321E5A"
MAGENTA = "#C4007A"
TEAL = "#009E8F"
ORANGE = "#E07A00"
SLATE = "#3F4756"
BAND = "#F1EDFA"
GREY = "#6B6B6B"
LINE = "#9A8FB5"

fig, ax = plt.subplots(figsize=(13.2, 7.1), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 72)
ax.axis("off")
ax.set_aspect("auto")


def band(y, h, label, color):
    ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=BAND, ec="none", zorder=1))
    tab = FancyBboxPatch((1.4, y + 0.6), 10.5, h - 1.2, boxstyle="round,pad=0.2,rounding_size=1.0",
                         fc=color, ec="none", zorder=2)
    ax.add_patch(tab)
    ax.text(6.65, y + h / 2, label, ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", rotation=90 if h > 10 else 0, zorder=3)


def box(x, y, w, h, title, sub=None, fc=PURPLE, tc="white", fs=11, sfs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=fc, ec="none", zorder=4))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=5)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color=tc,
                fontsize=sfs, zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=LINE, lw=2.2, zorder=3))


# Title
ax.text(50, 70, "Chest X-Ray Report Generation:  High-Level Design",
        ha="center", va="center", fontsize=16, fontweight="bold", color=PURPLE)

# ---- Bands -----------------------------------------------------------------
band(58.5, 8.5, "CLIENTS", SLATE)
band(44.5, 11.0, "API", PURPLE_D)
band(24.0, 15.5, "SERVICES", MAGENTA)
band(9.0, 11.0, "DATA", TEAL)

# ---- Clients ---------------------------------------------------------------
box(15, 60, 24, 5.5, "Gradio UI", "4 tabs: report · agent · KB · telemetry", fc=SLATE)
box(41.5, 60, 24, 5.5, "Swagger / OpenAPI", "interactive API docs at /docs", fc=SLATE)
box(68, 60, 24, 5.5, "REST / curl clients", "JSON over HTTPS", fc=SLATE)

# ---- API layer -------------------------------------------------------------
box(15, 51, 77, 3.4, "FastAPI + Uvicorn      middleware:  Request-ID  ·  Timing  ·  CORS  ·  Rate-limit",
    fc=PURPLE, fs=10.5)
box(15, 45.6, 24, 4.4, "system", "/health · /metadata · /metrics", fc="#6A4CA0", fs=10, sfs=8)
box(41.5, 45.6, 24, 4.4, "inference", "/predict · /predict/dicom · /task", fc="#6A4CA0", fs=10, sfs=8)
box(68, 45.6, 24, 4.4, "agent", "/agent/query · /search · /index", fc="#6A4CA0", fs=10, sfs=8)

# ---- Services --------------------------------------------------------------
box(15, 27, 18.5, 9.5, "Inference Engine", "DenseNet-121 encoder\n→ Transformer decoder", fc=MAGENTA, sfs=8)
box(35.7, 27, 18.5, 9.5, "DICOM Loader", "windowing:\nlung · mediastinum · bone", fc="#8E2C86", sfs=8)
box(56.4, 27, 18.5, 9.5, "RAG Agent", "ClinicalBERT embeds\nLangChain ReAct + LLM", fc=TEAL, sfs=8)
box(77.1, 27, 14.9, 9.5, "Async Worker", "Celery + Redis\n(queued jobs)", fc=ORANGE, sfs=8)

# ---- Data ------------------------------------------------------------------
box(15, 10.5, 24, 6, "Vector Store", "ChromaDB / FAISS", fc="#2F7D74")
box(41.5, 10.5, 24, 6, "Model Weights", "HuggingFace Hub", fc="#2F7D74")
box(68, 10.5, 24, 6, "Vocabulary", "vocab.json", fc="#2F7D74")

# ---- Arrows ----------------------------------------------------------------
for x in (27, 53.5, 80):
    arrow(x, 60, x, 55.0)          # clients -> api
arrow(27, 45.6, 24, 37.0)          # api -> inference engine
arrow(53.5, 45.6, 65, 37.0)        # api -> rag agent
arrow(80, 45.6, 84, 37.0)          # api -> async worker
arrow(65, 27, 68, 17.0)            # rag -> vector store
arrow(24, 27, 27, 17.0)            # engine -> model weights (approx)
arrow(27, 27, 50, 17.0)            # engine -> weights

# ---- Cross-cutting footer --------------------------------------------------
ax.add_patch(FancyBboxPatch((1, 1.5), 98, 5.5, boxstyle="round,pad=0.2,rounding_size=1.0",
                            fc=PURPLE, ec="none", zorder=2))
ax.text(28, 4.25, "Observability:  Prometheus  ·  OpenTelemetry  ·  MLflow",
        ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax.text(50, 4.25, "|", ha="center", va="center", color="#B9A6D8", fontsize=12)
ax.text(72, 4.25, "Delivery:  Docker  ·  Docker Compose  ·  GitHub Actions",
        ha="center", va="center", color="white", fontsize=10, fontweight="bold")

out = os.path.join(os.path.dirname(__file__), "hld_architecture.png")
plt.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("Saved:", out)
