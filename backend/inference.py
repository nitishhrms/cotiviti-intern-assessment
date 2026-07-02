import io
import json
import os
import time
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

logger = logging.getLogger("xray_api")


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
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
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3,
                 dropout=0.1, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=512, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, captions, image_features, tgt_mask=None, tgt_key_padding_mask=None):
        x = self.embedding(captions)
        x = self.pos_encoding(x)
        memory = image_features.unsqueeze(1)
        out = self.transformer_decoder(
            tgt=x, memory=memory,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)


class RadiologyReportModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3,
                 dropout=0.1, max_len=100):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TextDecoder(vocab_size, embed_dim, num_heads, num_layers, dropout, max_len)

    def forward(self, images, captions):
        seq_len = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        tgt_key_padding_mask = captions == 0
        image_features = self.encoder(images)
        return self.decoder(captions, image_features, tgt_mask, tgt_key_padding_mask)

    def generate(self, image, word2idx, idx2word, max_len=100, device="cpu"):
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            image_features = self.encoder(image)
            tokens = [word2idx["<start>"]]
            for _ in range(max_len):
                caption = torch.tensor([tokens], dtype=torch.long).to(device)
                logits = self.decoder(caption, image_features)
                next_token = logits[0, -1, :].argmax().item()
                tokens.append(next_token)
                if idx2word[next_token] == "<end>":
                    break
        words = [idx2word[t] for t in tokens
                 if idx2word[t] not in ("<start>", "<end>", "<pad>")]
        return " ".join(words)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "xray_caption_models"
VOCAB_PATH = BASE_DIR / "vocab.json"
CONFIG_PATH = MODEL_DIR / "config.json"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

MODEL_VERSION = "1.0.0"

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._load()

    def _load(self):
        try:
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)
            with open(VOCAB_PATH) as f:
                vocab_data = json.load(f)
            self.word2idx = vocab_data["word2idx"]
            self.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
            vocab_size = len(self.word2idx)

            self.model = RadiologyReportModel(
                vocab_size=vocab_size,
                embed_dim=self.config["embed_dim"],
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                dropout=self.config["dropout"],
                max_len=self.config["max_len"],
            ).to(self.device)

            if BEST_MODEL_PATH.exists():
                checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self._loaded = True
                logger.info("InferenceEngine: model loaded from %s", BEST_MODEL_PATH)
            else:
                logger.error("InferenceEngine: model file not found at %s", BEST_MODEL_PATH)
        except Exception as exc:
            logger.exception("InferenceEngine: failed to load model — %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, image_bytes: bytes) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded.")
        t0 = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = _TRANSFORM(image).to(self.device)
        report = self.model.generate(
            img_tensor, self.word2idx, self.idx2word,
            max_len=self.config["max_len"], device=self.device,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "report": report,
            "confidence": 1.0,
            "processing_time_ms": elapsed_ms,
            "model_version": MODEL_VERSION,
        }

    def predict_from_tensor(self, img_tensor: torch.Tensor) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded.")
        t0 = time.perf_counter()
        img_tensor = img_tensor.to(self.device)
        report = self.model.generate(
            img_tensor.squeeze(0), self.word2idx, self.idx2word,
            max_len=self.config["max_len"], device=self.device,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "report": report,
            "confidence": 1.0,
            "processing_time_ms": elapsed_ms,
            "model_version": MODEL_VERSION,
        }


engine = InferenceEngine()
