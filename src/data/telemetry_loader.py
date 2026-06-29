import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np

def load_and_scale_telemetry(cyber_path: str, physical_path: str) -> Tuple[np.ndarray, StandardScaler]:
    cyber_data = pd.read_csv(cyber_path)
    physical_data = pd.read_csv(physical_path)
    merged_telemetry = pd.concat([cyber_data, physical_data], axis=1)
    
    scaler = StandardScaler()
    scaled_telemetry = scaler.fit_transform(merged_telemetry.values)
    
    return scaled_telemetry, scaler
