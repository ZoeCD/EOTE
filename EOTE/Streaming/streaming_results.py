from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StreamingSampleResult:
    """Result for a single sample in the streaming window.

    Attributes:
        sample_index: Position of the sample within the window (0-indexed)
        score: Anomaly score computed by EOTE (positive = anomaly)
        is_outlier: True if score exceeds the configured outlier_threshold
        window_id: ID of the window this sample belongs to
        was_rescored: True if sample was rescored after model retraining
        original_score: Original score before retraining (None if not rescored)
    """
    sample_index: int
    score: float
    is_outlier: bool
    window_id: int
    was_rescored: bool = False
    original_score: Optional[float] = None


@dataclass
class WindowProcessingResult:
    """Result for a complete window processing operation.

    Attributes:
        window_id: Unique identifier for this window
        total_samples: Total number of samples in the window
        outlier_count: Number of samples classified as outliers
        outlier_ratio: Proportion of outliers (outlier_count / total_samples)
        retraining_triggered: True if model was retrained on this window
        training_samples_used: Number of samples used for retraining (0 if no retrain)
        samples_results: List of individual sample results in the window
        drift_detected: True if distribution drift was detected
        drift_p_value: KS test p-value for drift detection (None if not available)
        drift_ks_statistic: KS test statistic (None if not available)
    """
    window_id: int
    total_samples: int
    outlier_count: int
    outlier_ratio: float
    retraining_triggered: bool
    training_samples_used: int
    samples_results: List[StreamingSampleResult]
    drift_detected: bool = False
    drift_p_value: Optional[float] = None
    drift_ks_statistic: Optional[float] = None
