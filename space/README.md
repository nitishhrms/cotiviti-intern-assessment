---
title: Xray Report Generation + Clinical Agent
emoji: 🩻
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# Chest X-Ray Report Generation + Clinical RAG Agent

- **Report Generation** — DenseNet-121 encoder + Transformer decoder generate a radiology report from a chest X-ray.
- **Clinical Agent** — Retrieval-Augmented Generation (ClinicalBERT embeddings, cosine retrieval) with a ReAct reasoning trace.
- **Knowledge Base** — semantic search and index management over the report corpus.

Educational / research demo only. Not FDA-cleared for clinical use.

Source & docs: https://github.com/nitishhrms/cotiviti-intern-assessment
