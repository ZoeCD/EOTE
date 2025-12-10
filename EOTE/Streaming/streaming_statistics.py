from dataclasses import dataclass, field


@dataclass
class StreamingStatistics:
    """Tracks streaming EOTE operational metrics.

    All counters are cumulative across the lifetime of the StreamingEOTE instance.

    Attributes:
        windows_processed: Total number of complete windows processed
        retraining_count: Number of times model retraining was triggered and completed
        total_samples_processed: Total number of samples processed across all windows
        total_outliers_detected: Total number of samples classified as outliers
        retraining_skipped_count: Number of times retraining was triggered but skipped
            due to insufficient samples
        drift_detected_count: Number of windows where distribution drift was detected
    """
    windows_processed: int = 0
    retraining_count: int = 0
    total_samples_processed: int = 0
    total_outliers_detected: int = 0
    retraining_skipped_count: int = 0
    drift_detected_count: int = 0

    def record_window_processed(
        self,
        sample_count: int,
        outlier_count: int,
        retraining_triggered: bool,
        retraining_skipped: bool,
        drift_detected: bool = False
    ) -> None:
        """Update statistics after processing a window.

        Args:
            sample_count: Number of samples in the processed window
            outlier_count: Number of outliers detected in the window
            retraining_triggered: True if retraining was completed
            retraining_skipped: True if retraining was triggered but skipped
            drift_detected: True if distribution drift was detected
        """
        self.windows_processed += 1
        self.total_samples_processed += sample_count
        self.total_outliers_detected += outlier_count

        if retraining_triggered:
            self.retraining_count += 1

        if retraining_skipped:
            self.retraining_skipped_count += 1

        if drift_detected:
            self.drift_detected_count += 1

    def get_overall_outlier_ratio(self) -> float:
        """Calculate overall outlier ratio across all processed samples.

        Returns:
            Proportion of outliers (0.0 to 1.0), or 0.0 if no samples processed
        """
        if self.total_samples_processed == 0:
            return 0.0
        return self.total_outliers_detected / self.total_samples_processed

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.windows_processed = 0
        self.retraining_count = 0
        self.total_samples_processed = 0
        self.total_outliers_detected = 0
        self.retraining_skipped_count = 0
        self.drift_detected_count = 0
