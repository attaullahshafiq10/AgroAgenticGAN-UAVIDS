import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim

from configs.hyperparameters import Hyperparameters
from src.data.telemetry_loader import load_and_scale_telemetry
from src.data.sequence_tensor_builder import build_sliding_windows
from src.models.dual_gan_generator import DualGANGenerator
from src.models.dual_gan_discriminator import DualGANDiscriminator
from src.models.lstm_aerial_detector import LSTMAerialDetector, calculate_anomaly_score, evaluate_threat
from src.models.dqn_ground_agent import DQNGroundAgent, ExperienceReplayBuffer
from src.training.adversarial_trainer import execute_adversarial_training
from src.training.temporal_trainer import execute_temporal_training
from src.training.reinforcement_trainer import execute_reinforcement_training_step, select_epsilon_greedy_action, calculate_mitigation_reward
from src.utils.performance_metrics import evaluate_classification_metrics

def orchestrate_cyber_physical_pipeline():
    hyperparams = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dimension = 20
    synthetic_telemetry = np.random.randn(1000, feature_dimension).astype(np.float32)
    sequence_tensors = build_sliding_windows(synthetic_telemetry, hyperparams.sequence_length + 1)
    
    sequence_dataset = TensorDataset(torch.from_numpy(sequence_tensors))
    sequence_dataloader = DataLoader(sequence_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    
    flattened_telemetry_dataset = TensorDataset(torch.from_numpy(synthetic_telemetry))
    flattened_dataloader = DataLoader(flattened_telemetry_dataset, batch_size=hyperparams.batch_size, shuffle=True)

    generator = DualGANGenerator(hyperparams.latent_dim, feature_dimension).to(device)
    discriminator = DualGANDiscriminator(feature_dimension, hyperparams.dropout_rate).to(device)
    execute_adversarial_training(generator, discriminator, flattened_dataloader, hyperparams, device)

    lstm_detector = LSTMAerialDetector(
        feature_dimension, 
        hyperparams.hidden_layer_1_units, 
        hyperparams.hidden_layer_2_units, 
        hyperparams.dropout_rate
    ).to(device)
    execute_temporal_training(lstm_detector, sequence_dataloader, hyperparams, device)

    dqn_agent = DQNGroundAgent().to(device)
    target_dqn_agent = DQNGroundAgent().to(device)
    target_dqn_agent.load_state_dict(dqn_agent.state_dict())
    
    dqn_optimizer = optim.Adam(dqn_agent.parameters(), lr=0.001)
    replay_buffer = ExperienceReplayBuffer(hyperparams.replay_buffer_capacity)
    
    epsilon = hyperparams.epsilon_start
    moisture_state = 0.6
    tank_level_state = 0.8

    for episode in range(100):
        current_sequence = torch.randn(1, hyperparams.sequence_length, feature_dimension, device=device)
        with torch.no_grad():
            predicted_tensor = lstm_detector(current_sequence)
        
        ground_truth_tensor = torch.randn(1, feature_dimension, device=device)
        temporal_residual_error = calculate_anomaly_score(ground_truth_tensor, predicted_tensor)
        binary_threat_flag, threat_probability = evaluate_threat(temporal_residual_error, hyperparams.anomaly_threshold_tau)
        
        cyber_physical_state_vector = torch.tensor([moisture_state, tank_level_state, threat_probability.item()], dtype=torch.float32, device=device)
        
        for step in range(50):
            action = select_epsilon_greedy_action(dqn_agent, cyber_physical_state_vector, epsilon)
            reward = calculate_mitigation_reward(moisture_state, action)
            
            moisture_state = moisture_state + 0.1 if action == 1 else moisture_state - 0.05
            tank_level_state = tank_level_state - 0.1 if action == 1 else tank_level_state
            
            next_sequence = torch.randn(1, hyperparams.sequence_length, feature_dimension, device=device)
            with torch.no_grad():
                next_predicted_tensor = lstm_detector(next_sequence)
            next_temporal_residual_error = calculate_anomaly_score(torch.randn(1, feature_dimension, device=device), next_predicted_tensor)
            _, next_threat_probability = evaluate_threat(next_temporal_residual_error, hyperparams.anomaly_threshold_tau)
            
            next_cyber_physical_state_vector = torch.tensor([moisture_state, tank_level_state, next_threat_probability.item()], dtype=torch.float32, device=device)
            
            replay_buffer.append_transition(cyber_physical_state_vector, action, reward, next_cyber_physical_state_vector, False)
            cyber_physical_state_vector = next_cyber_physical_state_vector
            
            execute_reinforcement_training_step(dqn_agent, target_dqn_agent, dqn_optimizer, replay_buffer, hyperparams, device)
            
        epsilon = max(hyperparams.epsilon_min, epsilon * hyperparams.epsilon_decay)
        
        if episode % hyperparams.target_update_freq == 0:
            target_dqn_agent.load_state_dict(dqn_agent.state_dict())

if __name__ == "__main__":
    orchestrate_cyber_physical_pipeline()
