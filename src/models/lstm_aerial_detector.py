import torch
import torch.nn as nn
from typing import Tuple

class LSTMAerialDetector(nn.Module):
    def __init__(self, input_dimension: int, hidden_layer_1_units: int, hidden_layer_2_units: int, dropout_rate: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_layer_1_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        self.reconstruction_head = nn.Linear(hidden_layer_1_units, input_dimension)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(sequence_tensor)
        predicted_timestep = self.reconstruction_head(lstm_out[:, -1, :])
        return predicted_timestep

def calculate_anomaly_score(ground_truth: torch.Tensor, predicted_tensor: torch.Tensor) -> torch.Tensor:
    temporal_residual_error = torch.mean(torch.abs(ground_truth - predicted_tensor), dim=1)
    return temporal_residual_error

def evaluate_threat(temporal_residual_error: torch.Tensor, anomaly_threshold_tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    binary_threat_flag = (temporal_residual_error > anomaly_threshold_tau).float()
    threat_probability = torch.sigmoid(temporal_residual_error - anomaly_threshold_tau)
    return binary_threat_flag, threat_probability
