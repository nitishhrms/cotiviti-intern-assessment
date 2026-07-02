# Phase 6 — Uncertainty Quantification with Monte Carlo Dropout

## Goal

Enable the model to express *how confident it is* in its generated report by using Monte Carlo (MC) Dropout at inference time. Run N stochastic forward passes with dropout active, then aggregate results to produce per-token confidence scores and spatial uncertainty heatmaps. Flag low-confidence predictions with a clinician warning.

## Why This Matters for Your Resume

- Uncertainty quantification is a hard, unsolved problem in medical AI — very few candidates implement it
- Clinical AI regulations (FDA SaMD guidance) increasingly require models to express uncertainty
- MC Dropout is elegant: zero additional training required, just a change to inference behavior
- Token-level confidence scores are novel — most papers only do image-level uncertainty
- This is publishable-level research combined with Phase 4 (ViT+BioGPT+uncertainty)

---

## Prerequisites

- [ ] Phase 1 complete (Grad-CAM integrated — uncertainty heatmap will reuse this infrastructure)
- [ ] Phase 4 complete or in progress (ViT+BioGPT — uncertainty works better with larger models, but can start on baseline)
- [ ] Understanding of Bayesian deep learning basics (Gal & Ghahramani 2016)

---

## Task 1 — MC Dropout Inference Mode

### 1.1 — Enable Dropout at Inference
- [ ] Create `src/inference/mc_dropout.py`
- [ ] Implement `enable_mc_dropout(model)`:
  - [ ] Set the entire model to `eval()` mode (disables BatchNorm tracking)
  - [ ] Then re-enable all Dropout layers: iterate `model.modules()`, call `.train()` on `nn.Dropout` and `nn.Dropout2d` instances only
  - [ ] This is the key Gal & Ghahramani trick: eval mode for BN, train mode for Dropout
- [ ] Implement context manager `with mc_dropout_mode(model):` for clean usage

### 1.2 — Stochastic Forward Pass
- [ ] Implement `mc_forward_pass(model, image_tensor, n_samples=20) -> list[str]`:
  - [ ] Call `enable_mc_dropout(model)` before the loop
  - [ ] Run `n_samples` forward passes through the model
  - [ ] Collect N generated report strings
  - [ ] Return list of N strings (they will differ slightly due to stochastic dropout)

### 1.3 — Token-Level Probability Collection
- [ ] Implement `mc_token_probabilities(model, image_tensor, n_samples=20) -> dict`:
  - [ ] Instead of returning final decoded strings, collect raw logit distributions at each generation step
  - [ ] For each position `t` in the sequence, collect the softmax probability distribution from each of the N passes
  - [ ] Stack into tensor of shape `(n_samples, max_seq_len, vocab_size)`
  - [ ] Return this tensor alongside the N decoded strings

---

## Task 2 — Uncertainty Metrics Computation

### 2.1 — Predictive Entropy (Per-Token Uncertainty)
- [ ] Implement `compute_predictive_entropy(prob_tensor) -> np.ndarray`:
  - [ ] `prob_tensor`: shape `(n_samples, seq_len, vocab_size)`
  - [ ] Compute mean probability over samples: `p_mean = prob_tensor.mean(dim=0)` → `(seq_len, vocab_size)`
  - [ ] Compute entropy: `H = -sum(p_mean * log(p_mean + 1e-8), dim=-1)` → `(seq_len,)` — one value per token
  - [ ] Normalize to [0, 1] by dividing by `log(vocab_size)` (maximum possible entropy)
  - [ ] Returns: array of shape `(seq_len,)` — uncertainty score per generated token position

### 2.2 — Mutual Information (Epistemic Uncertainty)
- [ ] Implement `compute_mutual_information(prob_tensor) -> np.ndarray`:
  - [ ] Epistemic uncertainty = total uncertainty (predictive entropy) minus aleatoric uncertainty (expected entropy)
  - [ ] Expected entropy per sample: `E_H = mean(-sum(p * log(p + 1e-8), dim=-1), dim=0)` → `(seq_len,)`
  - [ ] MI = `H - E_H` → `(seq_len,)` — captures model uncertainty vs. data uncertainty
  - [ ] Higher MI = model is uncertain because it doesn't have enough information (not just noisy data)

### 2.3 — Report-Level Summary Score
- [ ] Implement `compute_report_confidence(token_entropies) -> float`:
  - [ ] Mean entropy over all non-special tokens → lower is more confident
  - [ ] Convert to a [0, 1] confidence score: `confidence = 1 - mean_entropy`
  - [ ] Threshold: if `confidence < 0.5`, flag for clinician review

### 2.4 — Report Consistency Score
- [ ] Implement `compute_report_consistency(n_reports: list[str]) -> float`:
  - [ ] Compute pairwise BLEU-1 scores between all N generated reports
  - [ ] Average pairwise score = consistency (1.0 = all identical, 0.0 = completely different)
  - [ ] Inconsistent reports across samples → model is not sure about the right words to use

---

## Task 3 — Spatial Uncertainty Heatmap

### 3.1 — Per-Region Uncertainty from Gradients
- [ ] Extend Grad-CAM from Phase 1 to compute *uncertainty-weighted* activation maps:
  - [ ] Run N MC Dropout passes and collect N Grad-CAM maps
  - [ ] Compute std deviation across N maps: `uncertainty_map = std(gradcam_maps, axis=0)`
  - [ ] High std deviation in a region = model is uncertain about what it sees there
  - [ ] Normalize `uncertainty_map` to [0, 1]

### 3.2 — Uncertainty Heatmap Visualization
- [ ] Implement `generate_uncertainty_heatmap(uncertainty_map, original_image) -> np.ndarray`:
  - [ ] Apply `cv2.COLORMAP_COOL` (blue = certain, red = uncertain) — different from Grad-CAM's COLORMAP_JET
  - [ ] Overlay on original image with alpha=0.5
  - [ ] Return: uncertainty overlay image

### 3.3 — Dual Heatmap Display
- [ ] Update UI to show three panels:
  - [ ] Panel 1: Original X-ray
  - [ ] Panel 2: Grad-CAM (what drove the report) — from Phase 1
  - [ ] Panel 3: Uncertainty heatmap (where the model is unsure) — new
- [ ] These two maps are complementary: a region can have high Grad-CAM activation *and* high uncertainty

---

## Task 4 — Aggregate Report Generation

### 4.1 — Most Common Token Selection
- [ ] After collecting N reports, select the "consensus report":
  - [ ] At each position, take the token with highest mean probability across N passes (mode voting)
  - [ ] This is more robust than a single greedy pass
- [ ] Implement `consensus_report(prob_tensor, tokenizer) -> str`

### 4.2 — Confidence-Annotated Report
- [ ] Implement `annotate_report_with_confidence(report_tokens, token_entropies) -> str`:
  - [ ] Low confidence tokens (entropy > threshold): wrap with `[UNCERTAIN: {word}]` marker
  - [ ] Example output: `"The lungs are clear. [UNCERTAIN: No] pleural effusion. Cardiac silhouette [UNCERTAIN: mildly] enlarged."`
  - [ ] This lets the radiologist see exactly which words the model is unsure about

---

## Task 5 — Calibration Analysis

### 5.1 — Reliability Diagram
- [ ] Implement `compute_calibration(predicted_probs, ground_truth_labels) -> dict`:
  - [ ] Bin predictions into 10 confidence bins: [0, 0.1], [0.1, 0.2], ..., [0.9, 1.0]
  - [ ] For each bin, compute average confidence and average accuracy
  - [ ] A well-calibrated model has these match (confidence = accuracy)
  - [ ] Compute Expected Calibration Error (ECE): weighted mean of |confidence - accuracy|

### 5.2 — Temperature Scaling
- [ ] If model is poorly calibrated (ECE > 0.05):
  - [ ] Implement post-hoc temperature scaling: divide logits by scalar T
  - [ ] Optimize T on the validation set to minimize negative log-likelihood
  - [ ] This is the standard calibration fix — fast, no retraining needed

### 5.3 — Calibration Plot
- [ ] Generate reliability diagram and save to `docs/figures/calibration.png`
- [ ] Include ECE before and after temperature scaling in `docs/results.md`

---

## Task 6 — UI Integration

### 6.1 — Confidence Display
- [ ] Add a confidence meter to Gradio: `gr.Number(label="Report Confidence", precision=1)` showing percentage
- [ ] Add colored badge: green (>80%), yellow (50–80%), red (<50%)
- [ ] Add `gr.Slider` for `n_samples` (default 20, range 5–50) — faster vs. more accurate uncertainty
- [ ] Show estimated processing time: `n_samples * single_inference_time`

### 6.2 — Clinician Warning
- [ ] If `confidence < 0.5` OR `consistency < 0.6`:
  - [ ] Show a `gr.Warning` box: "Low confidence prediction — radiologist review recommended"
  - [ ] This is a critical safety feature for clinical AI deployment

### 6.3 — Optional: Confidence-Annotated Report
- [ ] Add a checkbox: "Show token-level confidence annotation"
- [ ] When checked, display the annotated report with `[UNCERTAIN: ...]` markers
- [ ] When unchecked, show the clean consensus report

---

## Task 7 — Performance Optimization

### 7.1 — Batched MC Inference
- [ ] Instead of N sequential passes, run N copies of the same image in a batch: `image_batch = image.repeat(n_samples, 1, 1, 1)`
- [ ] This parallelizes the N forward passes — much faster on GPU
- [ ] Memory cost: `n_samples * single_image_memory` — use `n_samples=10` for GPU, `n_samples=5` for CPU

### 7.2 — Caching Single-Pass Results
- [ ] Run greedy decoding first (fast), return it immediately
- [ ] Run MC uncertainty estimation in background thread
- [ ] Update UI with uncertainty results when background task completes

---

## Task 8 — Testing

- [ ] Create `tests/test_uncertainty.py`
- [ ] Test `enable_mc_dropout()` — verify model produces different outputs on repeated calls
- [ ] Test `compute_predictive_entropy()` — entropy of one-hot distribution should be 0; uniform should be `log(vocab_size)`
- [ ] Test `compute_report_consistency()` — identical reports should have consistency=1.0
- [ ] Test `annotate_report_with_confidence()` — low-confidence tokens correctly wrapped
- [ ] Test `compute_calibration()` on synthetic data

---

## File Structure After This Phase

```
src/
  inference/
    mc_dropout.py       # MC inference, uncertainty metrics
    decoding.py         # Updated with consensus generation
tests/
  test_uncertainty.py
docs/
  figures/
    calibration.png
```

---

## Definition of Done

- [ ] MC Dropout produces visibly different reports across N passes (can demonstrate in notebook)
- [ ] Confidence score correlates with report quality (manually verify on 20 test cases)
- [ ] Uncertainty heatmap highlights plausible uncertain regions (edges, ambiguous pathology)
- [ ] Clinician warning triggers correctly for low-confidence predictions
- [ ] ECE after temperature scaling is < 0.05
- [ ] Processing time for n_samples=20 is < 10 seconds on GPU, < 60 seconds on CPU

---

## References

- Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017)
- Nix & Weigend, "Estimating the Mean and Variance of the Target Probability Distribution" (ICNN 1994)
- Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
