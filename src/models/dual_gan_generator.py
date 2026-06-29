import torch
import torch.nn as nn

class DualGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_feature_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_feature_dimension),
            nn.Tanh()
        )

    def forward(self, noise_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(noise_tensor)
