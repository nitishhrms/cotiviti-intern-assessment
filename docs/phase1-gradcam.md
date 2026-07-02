# Phase 1 — Visual Explainability with Grad-CAM

## Goal

Add Gradient-weighted Class Activation Mapping (Grad-CAM) to the existing DenseNet-121 encoder so that the system can visually explain *which regions* of the X-ray influenced the generated report. This is the single most important feature for clinical trust and regulatory acceptance of AI in radiology.

## Why This Matters for Your Resume

- Explainability is a hard requirement in FDA-regulated medical AI (21st Century Cures Act)
- Shows you understand *why* models make predictions, not just that they do
- Makes the Gradio demo visually striking and immediately understandable to non-technical interviewers

---

## Prerequisites

- [ ] Measure baseline BLEU-4 and ROUGE-L on the test split before any changes (establish ground truth numbers to cite on resume)
- [ ] Install additional dependencies: `opencv-python`, `matplotlib`, `scipy`
- [ ] Add dependencies to `requirements.txt`

---

## Task 1 — Implement Grad-CAM for DenseNet-121

### 1.1 — Hook Registration
- [ ] Create `src/explainability/gradcam.py`
- [ ] Register a forward hook on the last DenseNet dense block (`features.denseblock4`) to capture intermediate activations
- [ ] Register a backward hook on the same layer to capture gradients flowing back from the loss
- [ ] Store activations and gradients as instance attributes on the hook object

### 1.2 — Gradient Computation
- [ ] Write a `compute_gradcam(image_tensor, target_token_idx)` function
  - [ ] Forward pass: run encoder on image, get image features
  - [ ] Run decoder to generate full sequence
  - [ ] Backward pass: call `.backward()` on the logit for the target token
  - [ ] Pool gradients over spatial dimensions (global average pooling across H×W)
  - [ ] Weight activation maps by pooled gradients
  - [ ] Sum weighted maps, apply ReLU (negative values do not contribute)
  - [ ] Normalize output to [0, 1] range

### 1.3 — Heatmap Generation
- [ ] Write `generate_heatmap(gradcam_map, original_image)` function
  - [ ] Resize Grad-CAM output from (7×7 for DenseNet) to (224×224) using bicubic interpolation
  - [ ] Apply `cv2.COLORMAP_JET` colormap to convert grayscale weights to color
  - [ ] Blend heatmap with original X-ray using alpha compositing (alpha=0.4 for heatmap, 0.6 for image)
  - [ ] Return both the overlay image and the raw heatmap

### 1.4 — Multi-Token Aggregation
- [ ] Write `aggregate_gradcam_over_sequence(image_tensor, generated_tokens)` function
  - [ ] Compute per-token Grad-CAM for every non-special token in the generated report
  - [ ] Sum all per-token maps (uniform weighting) to produce a "full-report" heatmap
  - [ ] Optionally support per-sentence aggregation for longer reports

---

## Task 2 — Attention Rollout Visualization (Bonus)

### 2.1 — Cross-Attention Weight Extraction
- [ ] Modify `TextDecoder` in `app.py` to optionally return cross-attention weights from all decoder layers
- [ ] Average attention weights across all 8 heads and 3 decoder layers
- [ ] Reshape attention map from sequence length → spatial grid (to align with image patches)

### 2.2 — Token-to-Region Mapping
- [ ] Write `visualize_token_attention(token_str, attention_weights, original_image)` function
  - [ ] Given a specific word token (e.g., "effusion"), extract its row from the attention weight matrix
  - [ ] Reshape the spatial attention back to image dimensions
  - [ ] Overlay as a heatmap on the X-ray
- [ ] Create a word-click interface concept (for future UI integration)

---

## Task 3 — Update Gradio UI

### 3.1 — UI Layout Changes
- [ ] Open `app.py` and extend the Gradio interface
- [ ] Add a second image output component labeled "Explainability Map"
- [ ] Add a slider for `heatmap_alpha` (blend strength) with default 0.5
- [ ] Add a checkbox "Show per-token breakdown" for future token-level views

### 3.2 — Inference Function Update
- [ ] Refactor `generate_report(image)` in `app.py` to also return the Grad-CAM overlay
- [ ] Return tuple: `(report_text, overlay_image)`
- [ ] Handle edge case: if Grad-CAM computation fails (e.g., no gradient), return original image

### 3.3 — UI Polish
- [ ] Add a text label beneath the heatmap: "Regions highlighted in red had the highest influence on the generated report."
- [ ] Add a collapsible section in Gradio with a brief explanation of Grad-CAM for clinical users
- [ ] Update the Gradio `examples` list to include 2–3 real test X-rays with expected outputs

---

## Task 4 — Testing

- [ ] Create `tests/test_gradcam.py`
- [ ] Test that hook registration does not break forward pass output
- [ ] Test that `compute_gradcam` returns a tensor with shape `(224, 224)`
- [ ] Test that `generate_heatmap` returns an RGB image of correct size
- [ ] Test inference still produces valid text after Grad-CAM integration
- [ ] Visual sanity check: generate and manually inspect heatmaps on 5 test images

---

## Task 5 — Documentation

- [ ] Add "Explainability" section to `README.md` with a screenshot of the Grad-CAM overlay
- [ ] Write docstrings for all functions in `src/explainability/gradcam.py`
- [ ] Add to `docs/` a short technical note explaining Grad-CAM methodology and its clinical relevance

---

## File Structure After This Phase

```
src/
  explainability/
    __init__.py
    gradcam.py          # GradCAM class and helper functions
    attention_viz.py    # Attention rollout visualization
tests/
  test_gradcam.py
app.py                  # Updated with Grad-CAM integration
```

---

## Definition of Done

- [ ] Running `python app.py` launches Gradio with two output panels (report + heatmap)
- [ ] Heatmap visually highlights clinically relevant regions (lung fields, cardiac border, etc.)
- [ ] No regression in report generation quality (BLEU-4 unchanged)
- [ ] All tests in `tests/test_gradcam.py` pass
- [ ] Screenshot added to README.md

---

## References

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- Chefer et al., "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021)
- FDA guidance on AI/ML-Based Software as a Medical Device (SaMD)
