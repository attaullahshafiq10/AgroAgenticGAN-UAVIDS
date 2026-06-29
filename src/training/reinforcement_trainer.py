import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.models.dqn_ground_agent import DQNGroundAgent, ExperienceReplayBuffer
from configs.hyperparameters import Hyperparameters

def calculate_mitigation_reward(moisture_level: float, action: int, moisture_threshold: float = 0.5) -> float:
    if moisture_level < moisture_threshold:
        return 10.0 if action == 1 else -50.0
    return -10.0 if action == 1 else 5.0

def select_epsilon_greedy_action(
    q_network: DQNGroundAgent,
    state_vector: torch.Tensor,
    epsilon: float,
    action_space: int = 2
) -> int:
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    with torch.no_grad():
        return q_network(state_vector).argmax().item()

def execute_reinforcement_training_step(
    q_network: DQNGroundAgent,
    target_q_network: DQNGroundAgent,
    optimizer: optim.Optimizer,
    replay_buffer: ExperienceReplayBuffer,
    hyperparams: Hyperparameters,
    device: torch.device
):
    if len(replay_buffer) < hyperparams.dqn_batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample_batch(hyperparams.dqn_batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        max_next_q_values = target_q_network(next_states).max(1)[0]
        temporal_difference_targets = rewards + hyperparams.gamma_discount * max_next_q_values * (1 - dones)

    huber_loss_function = nn.SmoothL1Loss()
    td_loss = huber_loss_function(current_q_values, temporal_difference_targets)

    optimizer.zero_grad()
    td_loss.backward()
    optimizer.step()
