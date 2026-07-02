# Phase 4 — ViT + BioGPT Architecture Upgrade

## Goal

Replace the DenseNet-121 encoder + custom Transformer decoder with a modern architecture: Vision Transformer (ViT) as the image encoder and BioGPT-Large as the biomedical language decoder. Use LoRA (Low-Rank Adaptation) to enable fine-tuning on consumer hardware. Run a rigorous ablation study to quantify improvement over the baseline.

## Why This Matters for Your Resume

- ViT is the current SOTA vision backbone — shows you are not stuck in 2018
- BioGPT is a domain-specific language model trained on 15M PubMed abstracts — shows domain awareness
- LoRA fine-tuning is the industry-standard technique for efficient LLM adaptation
- A rigorous ablation table with BLEU/ROUGE/CIDEr numbers is publishable-level evaluation
- Most candidates submit projects with zero evaluation rigor — this sets you apart

---

## Prerequisites

- [ ] Phase 3 complete (MLflow tracking available)
- [ ] GPU with at least 12GB VRAM (Google Colab A100 is sufficient)
- [ ] Install: `transformers`, `peft` (for LoRA), `accelerate`, `bitsandbytes` (for 8-bit quantization)
- [ ] Verify: `pip install transformers peft accelerate bitsandbytes`

---

## Task 1 — ViT Encoder Integration

### 1.1 — Load Pretrained ViT
- [ ] Create `src/models/vit_encoder.py`
- [ ] Load `google/vit-base-patch16-224` from HuggingFace:
  - [ ] Use `ViTModel.from_pretrained("google/vit-base-patch16-224")`
  - [ ] Disable the classification head (we only want the hidden states)
- [ ] Implement `ViTEncoder(nn.Module)`:
  - [ ] `forward(pixel_values)` → returns patch embeddings from last hidden state: shape `(batch, 197, 768)`
    - 197 = 196 image patches (14×14 grid) + 1 CLS token
  - [ ] Project 768-dim ViT embeddings to 256-dim to match decoder input: `Linear(768, 256)`
  - [ ] Return both CLS embedding (global) and patch embeddings (spatial) separately

### 1.2 — Preprocessing Update
- [ ] Use `ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")` for preprocessing
- [ ] This handles resizing to 224×224 and normalization using ViT's specific mean/std values
- [ ] Update `src/data/transforms.py` to support ViT preprocessing as an alternative to the ImageNet pipeline

### 1.3 — Freeze/Unfreeze Strategy
- [ ] Freeze all ViT layers initially (only train projection layer)
- [ ] After 5 epochs, unfreeze the last 4 ViT transformer blocks for fine-tuning
- [ ] Implement `set_encoder_trainable(model, layers_from_end)` utility function

---

## Task 2 — BioGPT Decoder Integration

### 2.1 — Load BioGPT
- [ ] Create `src/models/biogpt_decoder.py`
- [ ] Load `microsoft/BioGPT-Large` from HuggingFace:
  - [ ] `BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")`
  - [ ] `BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")`
- [ ] BioGPT is a causal LM (GPT-style) — it generates text autoregressively

### 2.2 — Cross-Attention Injection (Image Conditioning)
- [ ] BioGPT is a decoder-only model — it has no native cross-attention to image features
- [ ] Two strategies (implement both, evaluate):
  - **Strategy A — Prefix Tokens:** Project image patch embeddings to BioGPT's token space and prepend as "visual prefix tokens" before the text tokens. Simplest approach.
  - **Strategy B — Cross-Attention Adapter:** Insert lightweight cross-attention layers between every 4th BioGPT transformer block. More expressive but harder to train.
- [ ] Start with Strategy A; move to B if BLEU-4 does not improve vs. baseline

#### Strategy A — Prefix Token Implementation
- [ ] Create `VisualPrefixLayer(nn.Module)`:
  - [ ] Project ViT patch embeddings: `Linear(256, biogpt_hidden_size)` where `biogpt_hidden_size = 1024`
  - [ ] Output: `(batch, num_visual_tokens, 1024)` — use all 196 patch tokens or average pool to 16
  - [ ] Concatenate with text token embeddings: `[visual_tokens | text_tokens]`
  - [ ] Adjust attention mask to include visual tokens

### 2.3 — LoRA Fine-Tuning Setup
- [ ] Apply LoRA to BioGPT using the `peft` library:
  ```python
  from peft import get_peft_model, LoraConfig, TaskType
  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16,              # LoRA rank
      lora_alpha=32,     # Scaling factor
      target_modules=["q_proj", "v_proj"],  # Apply to attention projections
      lora_dropout=0.05,
      bias="none"
  )
  biogpt = get_peft_model(biogpt, lora_config)
  ```
- [ ] This adds ~2M trainable params vs. 1.5B total — only the LoRA adapters are updated
- [ ] Print trainable parameter count: `model.print_trainable_parameters()`

### 2.4 — Unified ViT+BioGPT Model
- [ ] Create `src/models/vit_biogpt_model.py`
- [ ] Implement `ViTBioGPTModel(nn.Module)`:
  - [ ] Combine `ViTEncoder` + `VisualPrefixLayer` + LoRA-wrapped BioGPT
  - [ ] `forward(pixel_values, input_ids, attention_mask)` for training
  - [ ] `generate(pixel_values, max_new_tokens=100)` for inference using BioGPT's `.generate()` method
  - [ ] Support beam search: `num_beams=4`, `no_repeat_ngram_size=3`, `length_penalty=1.0`
  - [ ] Support sampling: `do_sample=True, temperature=0.7, top_p=0.9`

---

## Task 3 — Training

### 3.1 — Training Script
- [ ] Create `src/training/train_vit_biogpt.py`
- [ ] Use HuggingFace `Trainer` API for standardized training:
  - [ ] Define `TrainingArguments` with fp16/bf16 mixed precision
  - [ ] Use `DataCollatorWithPadding` for batch tokenization
  - [ ] Implement custom `compute_metrics` for BLEU-4 and ROUGE-L
- [ ] Alternative: use `accelerate` library for multi-GPU training

### 3.2 — Training Configuration
- [ ] Batch size: 8 (larger than before due to ViT memory requirements)
- [ ] Learning rate: 5e-5 for LoRA adapters, 1e-6 for unfrozen ViT layers
- [ ] Epochs: 20 with early stopping (patience=5)
- [ ] Use `get_cosine_schedule_with_warmup` with 10% warmup steps
- [ ] Mixed precision: `fp16=True` on NVIDIA, `bf16=True` on A100/H100

### 3.3 — Memory Optimization
- [ ] Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- [ ] Use `torch.cuda.amp.autocast()` for automatic mixed precision
- [ ] If OOM: apply 8-bit quantization via `bitsandbytes` to BioGPT base weights

---

## Task 4 — Decoding Strategy Comparison

### 4.1 — Implement Multiple Decoding Strategies
- [ ] Create `src/inference/decoding.py`
- [ ] Implement and compare:
  - [ ] **Greedy decoding** (argmax at each step) — baseline, what current model uses
  - [ ] **Beam search** (k=4) — generally best BLEU scores
  - [ ] **Top-p sampling** (nucleus sampling, p=0.9) — more diverse outputs
  - [ ] **Temperature scaling** (T=0.7) — reduce repetition in greedy
- [ ] Run all strategies on the same 100 test images, compute BLEU-4 and ROUGE-L for each
- [ ] Save comparison table to `docs/results.md`

---

## Task 5 — Ablation Study

### 5.1 — Configurations to Evaluate
Run each configuration and record BLEU-4, ROUGE-L, CIDEr, and METEOR:

| Config | Encoder | Decoder | Decoding |
|---|---|---|---|
| Baseline | DenseNet-121 | Custom Transformer | Greedy |
| Enc-Upgrade | ViT-base | Custom Transformer | Greedy |
| Dec-Upgrade | DenseNet-121 | BioGPT (LoRA) | Greedy |
| Full-Upgrade | ViT-base | BioGPT (LoRA) | Greedy |
| Full+Beam | ViT-base | BioGPT (LoRA) | Beam-4 |

- [ ] Train and evaluate each configuration
- [ ] Log all runs in MLflow with `experiment_name="ablation_study"`
- [ ] Create comparison plot: bar chart of BLEU-4 across configurations
- [ ] Save to `docs/figures/ablation_bleu4.png`

### 5.2 — Results Documentation
- [ ] Create `docs/results.md` with full ablation table
- [ ] Include number of trainable parameters per configuration
- [ ] Include inference latency (ms) on CPU and GPU per configuration

---

## Task 6 — Model Comparison Demo

### 6.1 — Side-by-Side Inference UI
- [ ] Update Gradio UI to show outputs from both models simultaneously
- [ ] Left column: DenseNet-121 + Custom Transformer (baseline)
- [ ] Right column: ViT + BioGPT (new)
- [ ] Let users compare quality on their own uploaded X-rays

### 6.2 — Attention Visualization for ViT
- [ ] Visualize ViT's self-attention maps (attention rollout):
  - [ ] Extract attention weights from the last 4 ViT blocks
  - [ ] Average across heads, apply attention rollout (recursive multiplication)
  - [ ] Reshape 14×14 attention grid back to 224×224
  - [ ] Overlay on input image
- [ ] Compare ViT attention map vs. DenseNet Grad-CAM for the same image

---

## Task 7 — Testing

- [ ] Create `tests/test_vit_biogpt.py`
- [ ] Test `ViTEncoder.forward()` returns shape `(batch, 197, 768)`
- [ ] Test `VisualPrefixLayer.forward()` returns correct shape for concatenation
- [ ] Test `ViTBioGPTModel.generate()` returns a non-empty string
- [ ] Test LoRA parameters are correctly identified as trainable
- [ ] Test decoding strategies produce different outputs (sanity check)

---

## File Structure After This Phase

```
src/
  models/
    vit_encoder.py
    biogpt_decoder.py
    vit_biogpt_model.py
  training/
    train_vit_biogpt.py
  inference/
    decoding.py
tests/
  test_vit_biogpt.py
docs/
  results.md
  figures/
    ablation_bleu4.png
    vit_attention_vs_gradcam.png
```

---

## Definition of Done

- [ ] ViT+BioGPT model trains to completion without NaN loss or OOM on Colab A100
- [ ] Full+Beam configuration outperforms baseline on BLEU-4 and ROUGE-L
- [ ] Ablation table documented in `docs/results.md` with real numbers
- [ ] Side-by-side demo shows qualitatively better reports from new model
- [ ] All tests pass

---

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
- Luo et al., "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining" (Briefings in Bioinformatics 2022)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002)
