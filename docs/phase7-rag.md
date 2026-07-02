# Phase 7 — RAG-Augmented Report Generation

## Goal

Build a retrieval-augmented generation (RAG) system where, at inference time, the system retrieves the most visually similar X-ray cases from a vector database and uses those historical reports as few-shot context to condition the decoder. This grounds the generated report in real-world evidence and dramatically reduces hallucination — the #1 safety concern in medical LLMs.

## Why This Matters for Your Resume

- RAG is the dominant paradigm for production LLM systems in 2024–2025
- Medical RAG is a hot research area (used at health systems like Mayo Clinic and Stanford Health)
- Grounding outputs in retrieved evidence is exactly how FDA-compliant AI systems are expected to work
- Demonstrates expertise across: vector databases, embedding models, retrieval algorithms, and LLM conditioning
- This is publishable at MICCAI, CHIL, or ML4H — especially combined with uncertainty quantification

---

## Prerequisites

- [ ] Phase 3 complete (FastAPI backend, MLflow available)
- [ ] Phase 4 recommended (ViT+BioGPT — RAG works best with a capable decoder)
- [ ] Install: `faiss-cpu` (or `faiss-gpu`), `chromadb` (alternative), `sentence-transformers`, `numpy`
- [ ] Indiana University training set images accessible (to build the index)

---

## Task 1 — Build the Vector Index

### 1.1 — Image Embedding Extraction
- [ ] Create `src/retrieval/build_index.py`
- [ ] Load the trained image encoder (DenseNet-121 or ViT-base from Phase 4)
- [ ] Set encoder to `eval()` mode with `torch.no_grad()`
- [ ] Iterate over all training images:
  - [ ] Preprocess each image → tensor
  - [ ] Forward pass through encoder → get 256-dim embedding (or 768-dim for ViT)
  - [ ] Store embedding alongside image path and corresponding gold report
- [ ] Save all embeddings as `numpy` array: shape `(n_train, embed_dim)`
- [ ] Save metadata (image paths, report texts) as JSON: `data/retrieval_index_metadata.json`

### 1.2 — FAISS Index Construction
- [ ] Install FAISS: `pip install faiss-cpu`
- [ ] Implement `build_faiss_index(embeddings: np.ndarray, use_ivf=False) -> faiss.Index`:
  - [ ] L2-normalize all embeddings: `faiss.normalize_L2(embeddings)` — enables cosine similarity via inner product
  - [ ] For small datasets (<50K): use `faiss.IndexFlatIP` (exact inner product search)
  - [ ] For large datasets (>50K): use `faiss.IndexIVFFlat` with `nlist=100` clusters for faster approximate search
  - [ ] If using IVF: train the index: `index.train(embeddings)` before adding vectors
  - [ ] Add all embeddings: `index.add(embeddings)`
  - [ ] Save index: `faiss.write_index(index, "data/xray_faiss.index")`
- [ ] Document: index size (MB), build time, and number of vectors indexed

### 1.3 — ChromaDB Alternative (Optional)
- [ ] For a more production-ready and persistent solution, implement the same pipeline using `chromadb`:
  - [ ] Create a `chromadb.PersistentClient` backed by a local directory
  - [ ] Create a collection with `cosine` distance metric
  - [ ] Add embeddings with metadata (image path, report text, split)
  - [ ] ChromaDB handles persistence automatically; no manual save/load needed

---

## Task 2 — Retrieval Pipeline

### 2.1 — Query Encoder
- [ ] Create `src/retrieval/retriever.py`
- [ ] Implement `ImageRetriever` class:
  - [ ] `__init__`: load FAISS index, metadata JSON, and image encoder
  - [ ] `encode_query(image_tensor) -> np.ndarray`: encode query image to embedding, L2-normalize
  - [ ] `retrieve(image_tensor, k=3) -> list[dict]`: return top-K most similar cases
    - [ ] Each result: `{'image_path': str, 'report': str, 'similarity_score': float, 'rank': int}`

### 2.2 — Diversity-Aware Retrieval (MMR)
- [ ] Implement Maximal Marginal Relevance (MMR) to avoid retrieving 3 nearly identical cases:
  - [ ] Retrieve top 20 candidates from FAISS
  - [ ] Greedily select K=3 that maximize `λ * similarity_to_query - (1-λ) * max_similarity_to_already_selected`
  - [ ] Default `λ=0.7` (favor relevance over diversity)
- [ ] Compare MMR vs. plain top-K retrieval in ablation: does diversity help report quality?

### 2.3 — Retrieval Quality Evaluation
- [ ] For each test image, retrieve top-K training images
- [ ] Compute pathology overlap: do retrieved cases share the same pathology labels?
- [ ] Expected: high semantic similarity in embedding space should correlate with similar pathologies
- [ ] Log retrieval precision@K in MLflow

---

## Task 3 — Context-Conditioned Report Generation

### 3.1 — Context Formatting
- [ ] Create `src/retrieval/context_builder.py`
- [ ] Implement `format_retrieval_context(retrieved_cases: list[dict]) -> str`:
  - [ ] Format K retrieved reports as a structured prompt:
    ```
    Similar case 1 (similarity: 0.92): "The lungs are clear. No pleural effusion. Cardiac silhouette normal."
    Similar case 2 (similarity: 0.88): "Mild cardiomegaly. No acute pulmonary disease. Stable."
    Similar case 3 (similarity: 0.85): "No acute cardiopulmonary process."
    Based on the above similar cases and the provided X-ray, generate a radiology report:
    ```
  - [ ] Truncate each retrieved report to 50 tokens max to fit within context window

### 3.2 — Prefix-Based Conditioning (for BioGPT)
- [ ] Tokenize the formatted context string using BioGPT tokenizer
- [ ] Prepend context tokens to the visual prefix tokens from Phase 4:
  - [ ] Final input sequence: `[visual_tokens | context_tokens | <start>]`
  - [ ] Adjust `attention_mask` to include context tokens (not masked)
- [ ] Use BioGPT's `.generate()` with the extended input

### 3.3 — Conditioning for Custom Transformer Decoder (Baseline Model)
- [ ] For the DenseNet-121 + custom transformer baseline:
  - [ ] Encode the context string with the word-level tokenizer
  - [ ] Treat context tokens as additional "memory" for the cross-attention mechanism
  - [ ] Concatenate context token embeddings with image embedding in the memory tensor

---

## Task 4 — Ablation Study

### 4.1 — Retrieval Configurations to Compare
Run on test set, compute BLEU-4, ROUGE-L, CIDEr, F1-Radgraph:

| Config | Retrieval | K | Context |
|---|---|---|---|
| No-RAG | None | 0 | Image only |
| RAG-K1 | FAISS exact | 1 | 1 report |
| RAG-K3 | FAISS exact | 3 | 3 reports |
| RAG-K5 | FAISS exact | 5 | 5 reports |
| RAG-MMR | FAISS + MMR | 3 | 3 diverse reports |
| RAG-Oracle | Ground truth similar | 3 | 3 gold reports |

- [ ] "Oracle" setup (using actual ground truth similar cases) gives an upper bound on RAG benefit
- [ ] Document all results in `docs/results.md`

### 4.2 — Hallucination Reduction Analysis
- [ ] Identify 20 test cases where the No-RAG model hallucinated (generated pathology not present in gold report)
- [ ] Measure: does RAG reduce hallucination rate for these cases?
- [ ] Compute hallucination rate: fraction of generated words not in any reference for the same patient

---

## Task 5 — Retrieval-Augmented Explainability

### 5.1 — Retrieved Case Display in UI
- [ ] In the Gradio UI, add a "Supporting Evidence" section below the generated report
- [ ] Show the top-3 retrieved cases:
  - [ ] Thumbnail of the retrieved X-ray (if available locally)
  - [ ] The retrieved report text
  - [ ] Similarity score: "92% visually similar"
- [ ] This gives clinicians evidence for why the model said what it said

### 5.2 — Difference Highlighting
- [ ] Implement `highlight_differences(generated_report, retrieved_reports) -> str`:
  - [ ] Find words in generated report not present in any retrieved report (potentially novel findings)
  - [ ] Highlight these in the UI with yellow background
  - [ ] This surfaces cases where the model is going beyond its retrieved evidence

---

## Task 6 — Production Considerations

### 6.1 — Index Update Strategy
- [ ] Implement `update_index(new_image_tensor, new_report, image_path)`:
  - [ ] Encode new image → add to FAISS index
  - [ ] Append to metadata JSON
  - [ ] This enables continuous learning as new cases arrive
- [ ] Add `POST /index/add` endpoint to FastAPI for adding new cases

### 6.2 — Index Versioning
- [ ] Tag FAISS index files with a version: `xray_faiss_v1.index`, `xray_faiss_v2.index`
- [ ] Store version in API response: `{"retrieval_index_version": "v2", ...}`
- [ ] Log index version in MLflow as a parameter

### 6.3 — Latency Budget
- [ ] Profile retrieval latency separately from model inference:
  - [ ] FAISS top-K retrieval should be < 10ms for datasets < 100K
  - [ ] Context tokenization: < 5ms
  - [ ] Total added latency: < 20ms
- [ ] If latency is too high, switch from exact search to IVF approximate search

---

## Task 7 — Testing

- [ ] Create `tests/test_retrieval.py`
- [ ] Test `build_faiss_index()` with 100 random embeddings — index has correct size
- [ ] Test `ImageRetriever.retrieve()` returns K results with correct schema
- [ ] Test retrieved results have similarity scores in [0, 1]
- [ ] Test `format_retrieval_context()` produces correctly formatted string
- [ ] Test full RAG pipeline end-to-end: image → retrieve → generate → string output
- [ ] Test `update_index()` increases index size by 1

---

## File Structure After This Phase

```
src/
  retrieval/
    __init__.py
    build_index.py      # Index construction script
    retriever.py        # ImageRetriever class
    context_builder.py  # Context formatting
data/
  xray_faiss.index
  retrieval_index_metadata.json
tests/
  test_retrieval.py
```

---

## Definition of Done

- [ ] FAISS index built over all training images (Indiana University dataset)
- [ ] RAG-K3 configuration achieves higher BLEU-4 than No-RAG on test set
- [ ] Retrieved cases displayed in Gradio UI with thumbnails and similarity scores
- [ ] Retrieval latency < 20ms per inference
- [ ] Full ablation table documented in `docs/results.md`
- [ ] All tests pass

---

## References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- Johnson et al., "MIMIC-CXR: A Large Publicly Available Database of Labeled Chest Radiographs" (2019)
- Dalla Serra et al., "Finding-Aware Anatomical Tokens for Chest X-Ray Automated Reporting" (ECCV 2024)
- Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents" (SIGIR 1998)
- FAISS documentation: https://github.com/facebookresearch/faiss
