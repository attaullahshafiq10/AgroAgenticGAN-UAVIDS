import torch
import torch.nn as nn
import torch.optim as optim
from src.models.lstm_aerial_detector import LSTMAerialDetector
from configs.hyperparameters import Hyperparameters

def execute_temporal_training(
    lstm_detector: LSTMAerialDetector,
    dataloader: torch.utils.data.DataLoader,
    hyperparams: Hyperparameters,
    device: torch.device
):
    mse_loss = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_detector.parameters(), lr=hyperparams.lstm_lr)

    for epoch in range(hyperparams.lstm_epochs):
        for sequence_batch in dataloader:
            sequence_tensors = sequence_batch[0].to(device)
            
            input_sequences = sequence_tensors[:, :-1, :]
            ground_truth_timestep = sequence_tensors[:, -1, :]

            lstm_optimizer.zero_grad()
            predicted_timestep = lstm_detector(input_sequences)
            
            temporal_loss = mse_loss(predicted_timestep, ground_truth_timestep)
            temporal_loss.backward()
            lstm_optimizer.step()
