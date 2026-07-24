"""
Render a clean High-Level Design (HLD) architecture diagram as a PNG,
used in both the Word report and the PowerPoint deck.

The wiring mirrors the real FastAPI app in backend/main.py:

    tags=["system"]     /health, /metadata, /metrics      -> no service dependency
    tags=["inference"]  /predict, /predict/dicom,
                        /preview/dicom, /task/{task_id}   -> Inference Engine, DICOM Loader,
                                                             Celery worker (async)
    tags=["agent"]      /agent/query, /search,
                        /index, /stats                    -> RAG Agent

    Inference Engine -> model weights + vocab.json
    RAG Agent        -> vector store
    Celery worker    -> Redis (broker AND result backend, see backend/celery_app.py)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

PURPLE = "#4B2E83"
PURPLE_D = "#321E5A"
PURPLE_L = "#6A4CA0"
MAGENTA = "#C4007A"
PLUM = "#8E2C86"
TEAL = "#009E8F"
TEAL_D = "#2F7D74"
ORANGE = "#E07A00"
SLATE = "#3F4756"
BAND = "#F1EDFA"
LINE = "#7E6FA3"
ASYNC = "#E07A00"

fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 74)
ax.axis("off")
ax.set_aspect("auto")


def band(y, h, label, color):
    ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=BAND, ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((1.4, y + 0.6), 10.5, h - 1.2,
                                boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=color, ec="none", zorder=2))
    ax.text(6.65, y + h / 2, label, ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", zorder=3)


def box(x, y, w, h, title, sub=None, fc=PURPLE, tc="white", fs=11, sfs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=fc, ec="none", zorder=4))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=5)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=tc,
                fontsize=sfs, zorder=5, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=5)


def arrow(x1, y1, x2, y2, dashed=False, color=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        color=color or (ASYNC if dashed else LINE), lw=2.0,
        linestyle=(0, (4, 2.5)) if dashed else "solid", zorder=3))


def step(n, x, y):
    """Numbered badge so the diagram can be narrated in order."""
    ax.add_patch(Circle((x, y), 1.55, fc="white", ec=LINE, lw=1.6, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", color=PURPLE,
            fontsize=8.5, fontweight="bold", zorder=7)


# ---- Title + legend --------------------------------------------------------
ax.text(50, 72.1, "Chest X-Ray Report Generation:  High-Level Design",
        ha="center", va="center", fontsize=16, fontweight="bold", color=PURPLE)

ax.add_patch(FancyArrowPatch((33.5, 68.6), (37.5, 68.6), arrowstyle="-|>",
                             mutation_scale=13, color=LINE, lw=2.0))
ax.text(38.4, 68.6, "synchronous request path", ha="left", va="center",
        fontsize=8.5, color="#4A4A4A")
ax.add_patch(FancyArrowPatch((56.5, 68.6), (60.5, 68.6), arrowstyle="-|>",
                             mutation_scale=13, color=ASYNC, lw=2.0,
                             linestyle=(0, (4, 2.5))))
ax.text(61.4, 68.6, "asynchronous / background path", ha="left", va="center",
        fontsize=8.5, color="#4A4A4A")

# ---- Bands -----------------------------------------------------------------
band(58.0, 8.6, "CLIENTS", SLATE)
band(44.0, 11.2, "API", PURPLE_D)
band(23.5, 15.8, "SERVICES", MAGENTA)
band(8.8, 11.2, "DATA", TEAL)

# ---- Clients ---------------------------------------------------------------
box(14, 59.6, 24, 5.5, "Gradio UI", "4 tabs · local + HF Space", fc=SLATE)
box(40, 59.6, 26, 5.5, "Swagger / OpenAPI", "interactive API docs at /docs", fc=SLATE)
box(68, 59.6, 24, 5.5, "REST / curl clients", "JSON over HTTPS", fc=SLATE)

# ---- API layer -------------------------------------------------------------
box(14, 50.6, 78, 3.4,
    "FastAPI + Uvicorn      middleware:  Request-ID  ·  Timing  ·  CORS  ·  Rate-limit",
    fc=PURPLE, fs=10.5)
box(14, 45.2, 24, 4.4, "system", "/health · /metadata · /metrics",
    fc=PURPLE_L, fs=10, sfs=8)
box(40, 45.2, 26, 4.4, "inference", "/predict · /predict/dicom · /task",
    fc=PURPLE_L, fs=10, sfs=8)
box(68, 45.2, 24, 4.4, "agent", "/agent/query · /search · /index",
    fc=PURPLE_L, fs=10, sfs=8)

# ---- Services --------------------------------------------------------------
# Column order is chosen so that every arrow below runs without crossing another.
box(14, 26.5, 15, 9.8, "DICOM Loader",
    "windowing:\nlung · bone", fc=PLUM, fs=10, sfs=7.5)
box(30.5, 26.5, 15, 9.8, "Inference Engine",
    "DenseNet-121\n→ Transformer", fc=MAGENTA, fs=10, sfs=7.5)
box(47, 26.5, 15, 9.8, "Report Formatter",
    "sections +\nprovenance", fc="#A32C6D", fs=10, sfs=7.5)
box(63.5, 26.5, 15, 9.8, "Async Worker",
    "Celery ·\n/task/{id}", fc=ORANGE, fs=10, sfs=7.5)
box(80, 26.5, 15, 9.8, "RAG Agent",
    "ClinicalBERT +\ncosine + ReAct", fc=TEAL, fs=10, sfs=7.5)

# ---- Data ------------------------------------------------------------------
box(14, 10.3, 15, 6.0, "Model Weights", "HF Hub", fc=TEAL_D, fs=10, sfs=8)
box(30.5, 10.3, 15, 6.0, "Vocabulary", "vocab.json", fc=TEAL_D, fs=10, sfs=8)
box(47, 10.3, 15, 6.0, "PDF Report", "downloadable", fc=TEAL_D, fs=10, sfs=8)
box(63.5, 10.3, 15, 6.0, "Redis", "broker + results", fc=TEAL_D, fs=10, sfs=8)
box(80, 10.3, 15, 6.0, "Vector Store", "ChromaDB / FAISS", fc=TEAL_D, fs=10, sfs=7.5)

# ---- Arrows: clients -> API ------------------------------------------------
for x in (26, 53, 80):
    arrow(x, 59.6, x, 54.3)
step(1, 26, 57.0)

# ---- Arrows: API -> services -----------------------------------------------
arrow(49, 45.2, 22.5, 36.8)              # inference -> DICOM Loader
arrow(53, 45.2, 38, 36.8)                # inference -> Inference Engine
arrow(57, 45.2, 70, 36.8, dashed=True)   # inference -> Async Worker (enqueue)
arrow(80, 45.2, 86, 36.8)                # agent     -> RAG Agent
step(2, 47.5, 42.4)
step(5, 83.4, 41.2)

# Service-to-service hand-offs (adjacent boxes)
arrow(29, 31.4, 30.2, 31.4)              # DICOM Loader -> Inference Engine
arrow(45.5, 31.4, 46.7, 31.4)            # Inference Engine -> Report Formatter
step(3, 46.1, 31.4)

# ---- Arrows: services -> data ----------------------------------------------
arrow(34, 26.5, 23, 16.6)                # engine    -> model weights
arrow(38, 26.5, 38, 16.6)                # engine    -> vocabulary
arrow(54.5, 26.5, 54.5, 16.6)            # formatter -> PDF report
arrow(71, 26.5, 71, 16.6, dashed=True)   # worker    -> redis
arrow(87.5, 26.5, 87.5, 16.6)            # RAG agent -> vector store
step(4, 54.5, 21.6)
step(6, 87.5, 21.6)

# ---- Cross-cutting footer --------------------------------------------------
ax.add_patch(FancyBboxPatch((1, 1.4), 98, 5.4, boxstyle="round,pad=0.2,rounding_size=1.0",
                            fc=PURPLE, ec="none", zorder=2))
ax.text(4.5, 4.1, "Observability:  Prometheus  ·  OpenTelemetry  ·  MLflow",
        ha="left", va="center", color="white", fontsize=9.5, fontweight="bold")
ax.text(53, 4.1, "|", ha="center", va="center", color="#B9A6D8", fontsize=12)
ax.text(95.5, 4.1, "Delivery:  Docker  ·  Docker Compose  ·  GitHub Actions",
        ha="right", va="center", color="white", fontsize=9.5, fontweight="bold")

out = os.path.join(os.path.dirname(__file__), "hld_architecture.png")
plt.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print("Saved:", out)
