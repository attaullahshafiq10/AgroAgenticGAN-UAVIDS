import numpy as np
from typing import Dict

def evaluate_classification_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]:
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))
    
    total_samples = true_positives + false_positives + true_negatives + false_negatives
    
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "true_negatives": float(true_negatives),
        "false_negatives": float(false_negatives),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
