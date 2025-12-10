from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """Configuration for streaming EOTE behavior.

    Attributes:
        window_size: Number of samples per window (non-overlapping).
            Must be positive. Default: 100
        outlier_threshold: Score threshold for classifying outliers.
            Samples with score > threshold are considered outliers. Default: 0.0
        drift_significance_level: P-value threshold for KS test drift detection.
            Drift detected when p-value < this threshold. Default: 0.05 (95% confidence)
        retraining_percentile: Proportion of lowest scores to keep for retraining.
            Keeps bottom N% of scores (e.g., 0.75 = keep 75%). Default: 0.75
        min_normal_samples: Minimum number of samples required to
            trigger retraining. Prevents retraining on insufficient data. Default: 10
        initial_training_required: If True, train() must be called before
            processing samples. If False, first window can be used for initial training. Default: True
    """
    window_size: int = 100
    outlier_threshold: float = 0.0
    drift_significance_level: float = 0.05
    retraining_percentile: float = 0.75
    min_normal_samples: int = 10
    initial_training_required: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not 0.0 < self.drift_significance_level < 1.0:
            raise ValueError(
                f"drift_significance_level must be between 0.0 and 1.0, "
                f"got {self.drift_significance_level}"
            )
        if not 0.0 < self.retraining_percentile < 1.0:
            raise ValueError(
                f"retraining_percentile must be between 0.0 and 1.0, "
                f"got {self.retraining_percentile}"
            )
        if self.min_normal_samples < 1:
            raise ValueError(
                f"min_normal_samples must be at least 1, got {self.min_normal_samples}"
            )
        if self.min_normal_samples > self.window_size:
            raise ValueError(
                f"min_normal_samples ({self.min_normal_samples}) cannot exceed "
                f"window_size ({self.window_size})"
            )
