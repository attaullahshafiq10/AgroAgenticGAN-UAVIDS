import numpy as np

def build_sliding_windows(telemetry_data: np.ndarray, sequence_length: int) -> np.ndarray:
    num_samples = telemetry_data.shape[0] - sequence_length + 1
    feature_dimension = telemetry_data.shape[1]
    
    shape = (num_samples, sequence_length, feature_dimension)
    strides = (telemetry_data.strides[0], telemetry_data.strides[0], telemetry_data.strides[1])
    
    sequence_tensors = np.lib.stride_tricks.as_strided(telemetry_data, shape=shape, strides=strides)
    return sequence_tensors
