from dataclasses import dataclass

@dataclass
class Hyperparameters:
    t_its_dataset_path: str = "./data/T-ITS_Dataset.csv"
    acwa_dataset_path: str = "./data/ACWA_balanced_dataset.csv"
    
    latent_dim: int = 100
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    gan_epochs: int = 500
    batch_size: int = 64
    
    hidden_layer_1_units: int = 128
    hidden_layer_2_units: int = 64
    sequence_length: int = 20
    dropout_rate: float = 0.3
    lstm_lr: float = 0.001
    lstm_epochs: int = 50
    anomaly_threshold_tau: float = 0.75
    
    gamma_discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 10
    replay_buffer_capacity: int = 100000
    dqn_batch_size: int = 64
