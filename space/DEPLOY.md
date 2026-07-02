# Hugging Face Space (Live Demo)

This folder contains the **self-contained** app that is deployed to the live demo Space:

**https://huggingface.co/spaces/nitishhrms/xray-report-generation**

It is a single-file Gradio app (no `backend/` import needed) so it runs on a free
CPU Space. It has four tabs:

1. **Report Generation** — DenseNet-121 encoder + Transformer decoder.
2. **Clinical Agent** — RAG: ClinicalBERT embeddings + cosine retrieval + ReAct trace
   (falls back to a TF-IDF retriever if ClinicalBERT cannot load, so the tab always works).
3. **Knowledge Base** — semantic search + add-report (index management).
4. **About**.

## Redeploy

```bash
# files pushed to the Space repo: app.py, requirements.txt, README.md
huggingface-cli upload nitishhrms/xray-report-generation space/app.py app.py --repo-type space
```

> The trained weights (`xray_caption_models/best_model.pth`) and `vocab.json` already
> live in the Space repo and are not duplicated here.
