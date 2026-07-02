---
language: en
license: mit
pipeline_tag: image-to-text
tags:
  - medical
  - radiology
  - chest-xray
  - report-generation
  - densenet
  - transformer
datasets:
  - indiana-university-chest-xray
metrics:
  - bleu
  - rouge
---

# X-Ray Report Generator

Automated chest X-ray report generation using a DenseNet-121 encoder and a
Transformer decoder, trained on the Indiana University Chest X-ray Collection.

## Model Description

| Component | Architecture |
|-----------|-------------|
| Image Encoder | DenseNet-121 (ImageNet pre-trained) + Linear projection (256-d) |
| Text Decoder | 3-layer Transformer Decoder, 8 heads, 256-d embeddings |
| Training data | Indiana University Chest X-ray Collection (~3,900 studies) |

## Intended Use

- **Primary use**: Educational and research demonstration of automated radiology AI.
- **Out of scope**: Clinical diagnosis. This model is **not** FDA-cleared and must
  not be used for patient care without clinical validation.

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| BLEU-4 | _see MLflow run_ |
| ROUGE-L | _see MLflow run_ |

## How to Use

```python
from huggingface_hub import hf_hub_download
import torch, json
from PIL import Image
from torchvision import transforms

# Download artifacts
model_path = hf_hub_download(repo_id="your-username/xray-report-generator", filename="best_model.pth")
vocab_path  = hf_hub_download(repo_id="your-username/xray-report-generator", filename="vocab.json")
config_path = hf_hub_download(repo_id="your-username/xray-report-generator", filename="config.json")

# Load and run — see app.py for full inference code
```

## Limitations and Bias

- Trained exclusively on a US academic hospital dataset. Performance may degrade
  on images from different scanner manufacturers or patient populations.
- Does not replace a qualified radiologist.
- Patient age/sex conditioning is forward-compatible but requires retraining to activate.

## Ethical Considerations

- No patient-identifiable information is included in the training data.
- The model should only be deployed in contexts where a radiologist reviews outputs.
- Downstream users must comply with applicable medical device regulations.

## Citation

Indiana University Chest X-Ray Collection — Demner-Fushman et al., 2016.
