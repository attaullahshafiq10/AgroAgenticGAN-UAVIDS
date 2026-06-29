import torch
import torch.nn as nn

class DualGANDiscriminator(nn.Module):
    def __init__(self, input_feature_dimension: int, dropout_probability: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_feature_dimension, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_probability),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_probability),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(feature_tensor)
