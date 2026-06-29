import torch
import torch.nn as nn
import collections
import random
from typing import Tuple

class DQNGroundAgent(nn.Module):
    def __init__(self, cyber_physical_state_dimension: int = 3, action_space: int = 2):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(cyber_physical_state_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, cyber_physical_state_vector: torch.Tensor) -> torch.Tensor:
        return self.q_network(cyber_physical_state_vector)

class ExperienceReplayBuffer:
    def __init__(self, replay_buffer_capacity: int):
        self.memory = collections.deque(maxlen=replay_buffer_capacity)

    def append_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.stack(next_states), torch.tensor(dones, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.memory)
