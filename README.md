# AgroAgenticGAN

Welcome to the official repository for **AgroAgenticGAN**, a two-tier cyber-physical framework designed for secure anomaly detection and autonomous control in UAV-assisted smart irrigation networks.

This architecture bridges the cyber-physical divide by combining an Aerial Tier for high-speed network intrusion detection with a Ground Tier governing physical actuation. 

Our core components include:
- **Dual-GAN**: A generative data augmentation module that resolves extreme class imbalances by oversampling minority attack distributions.
- **Deep LSTM**: A sequential detector that processes continuous cyber-physical telemetry to identify anomalous patterns and output threat confidence scores.
- **Deep Q-Network (DQN)**: A reinforcement learning agent enforcing closed-loop smart irrigation policies, dynamically mitigating false positives without resource penalization.

## Directory Structure
```text
/agroagenticgan_research
├── configs/               # Centralized hyperparameters
├── dataset/               # T-ITS and ACWA benchmark datasets
├── src/
│   ├── data/              # Telemetry loading and sequential window builders
│   ├── models/            # Dual-GAN, LSTM, and DQN architectures
│   ├── training/          # Decoupled optimization loops
│   └── utils/             # Performance metrics and explainability (SHAP/LIME)
├── requirements.txt       # Project dependencies
└── main_pipeline.py       # Unified execution script
```

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/attaullahshafiq10/AgroAgenticGAN-UAVIDS.git
cd agroagenticgan_research
pip install -r requirements.txt
```

### 2. Datasets
Ensure your datasets are placed in the `./data/` directory (or update paths in `configs/hyperparameters.py`). 

This framework utilizes the following datasets for its cyber-physical evaluation. If you use this code in your research, please cite the original datasets:

**T-ITS UAV Dataset**
```bibtex
@article{T-ITS,
  author       = {Hassler, Samuel Chase and Mughal, Umair Ahmad and Ismail, Muhammad},
  title        = {Cyber-Physical Intrusion Detection System for Unmanned Aerial Vehicles},
  journal      = {IEEE Transactions on Intelligent Transportation Systems},
  volume       = {25},
  number       = {6},
  pages        = {6106--6117},
  year         = {2024},
  doi          = {10.1109/TITS.2023.3339728},
  publisher    = {IEEE}
}
```

**ACWA Water Systems Dataset**
```bibtex
@article{ACWA,
    author = {Batarseh, Feras A. and Kulkarni, Ajay and Sreng, Chhayly and Lin, Justice and Maksud, Siam},
    title = {ACWA: an AI-driven cyber-physical testbed for intelligent water systems},
    journal = {Water Practice and Technology},
    volume = {18},
    number = {12},
    pages = {3399-3418},
    year = {2023},
    doi = {10.2166/wpt.2023.197}
}
```

### 3. Execution
Run the unified pipeline, which sequentially handles data augmentation, temporal training, and DQN agent optimization:
```bash
python main_pipeline.py
```

# AgroAgenticGAN-UAVIDS