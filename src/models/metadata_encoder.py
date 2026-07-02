import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.age_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        self.sex_embedding = nn.Embedding(3, 64)
        self.fusion = nn.Linear(128, embed_dim)

    def forward(self, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        age_emb = self.age_fc(age.unsqueeze(-1).float())
        sex_emb = self.sex_embedding(sex.long())
        combined = torch.cat([age_emb, sex_emb], dim=-1)
        return self.fusion(combined)
