import pandas as pd
import numpy as np
from typing import Optional, List
from scipy.stats import ks_2samp
from EOTE.EOTE import EOTE
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from .window_buffer import WindowBuffer
from .streaming_config import StreamingConfig
from .streaming_statistics import StreamingStatistics
from .streaming_logger import StreamingLogger, DefaultStreamingLogger
from .streaming_results import StreamingSampleResult, WindowProcessingResult


class StreamingEOTE:
    """Streaming wrapper for EOTE outlier detection with adaptive retraining.

    StreamingEOTE processes data in fixed-size, non-overlapping windows. When
    distribution drift is detected using the Kolmogorov-Smirnov test, the model
    is retrained on the bottom N% of scores (adaptive percentile filtering) from
    that window and all samples are re-scored.

    This enables adaptation to concept drift in streaming data while maintaining
    the interpretability of EOTE's decision tree-based approach.

    Example:
        >>> from EOTE import StreamingEOTE
        >>> streaming = StreamingEOTE(window_size=100, outlier_threshold=0.0)
        >>> streaming.train(X_train, y_train)
        >>> for sample in data_stream.iterrows():
        ...     result = streaming.process_sample(sample[1])
        ...     if result and result.retraining_triggered:
        ...         print(f"Drift detected! Model retrained on window {result.window_id}")
    """

    def __init__(
        self,
        eote: Optional[EOTE] = None,
        window_size: int = 100,
        outlier_threshold: float = 0.0,
        drift_significance_level: float = 0.05,
        retraining_percentile: float = 0.75,
        min_normal_samples: int = 10,
        initial_training_required: bool = True,
        logger: Optional[StreamingLogger] = None
    ):
        """Initialize StreamingEOTE.

        Args:
            eote: Pre-configured EOTE instance. If None, creates a default EOTE
                using EoteWithMissForestInTerminalBuilder.
            window_size: Number of samples per window (default: 100)
            outlier_threshold: Score above which samples are considered outliers (default: 0.0)
            drift_significance_level: P-value threshold for KS test drift detection (default: 0.05)
            retraining_percentile: Keep bottom N% of scores for retraining (default: 0.75)
            min_normal_samples: Minimum samples required for retraining (default: 10)
            initial_training_required: If True, train() must be called before processing (default: True)
            logger: Custom logger. If None, uses DefaultStreamingLogger(verbose=True)
        """
        # Build default EOTE if not provided
        if eote is None:
            director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
            eote = director.get_eote()

        self._eote = eote
        self._config = StreamingConfig(
            window_size=window_size,
            outlier_threshold=outlier_threshold,
            drift_significance_level=drift_significance_level,
            retraining_percentile=retraining_percentile,
            min_normal_samples=min_normal_samples,
            initial_training_required=initial_training_required
        )
        self._buffer = WindowBuffer(window_size)
        self._statistics = StreamingStatistics()
        self._logger = logger if logger is not None else DefaultStreamingLogger(verbose=True)
        self._is_trained = False
        self._baseline_scores = None  # Will be set during training

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform initial training on labeled normal data.

        This method must be called before processing streaming samples if
        initial_training_required is True.

        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
            y: Target Series with shape (n_samples,). Should contain only one class
               (typically 'normal') for semi-supervised anomaly detection.

        Raises:
            ValueError: If EOTE training fails (e.g., multiple classes, insufficient data)
        """
        self._eote.train(X, y)
        self._is_trained = True

        # Baseline scores will be set from the first processed window
        # to avoid issues with training data being different from streaming data
        self._baseline_scores = None

        self._logger.log_info("Initial training completed")

    def process_sample(self, sample: pd.Series) -> Optional[WindowProcessingResult]:
        """Process a single sample in the streaming pipeline.

        Adds the sample to the current window buffer. When the window becomes full,
        processes the entire window (scoring, outlier detection, potential retraining).

        Args:
            sample: pandas Series representing one instance with features as index

        Returns:
            WindowProcessingResult if a complete window was processed, None otherwise

        Raises:
            ValueError: If model is not trained and initial_training_required is True
        """
        self._check_if_trained()

        # Add sample to buffer
        window_full = self._buffer.add_sample(sample)

        # Process window if full
        if window_full:
            return self._process_window()

        return None

    def process_batch(self, samples: pd.DataFrame) -> List[WindowProcessingResult]:
        """Process multiple samples efficiently.

        Processes samples one at a time, collecting WindowProcessingResults when
        windows complete. This is more efficient than calling process_sample()
        repeatedly.

        Args:
            samples: DataFrame containing multiple samples (rows)

        Returns:
            List of WindowProcessingResults for each completed window.
            May be empty if no windows were completed.

        Raises:
            ValueError: If model is not trained and initial_training_required is True
        """
        self._check_if_trained()

        results = []
        for idx, sample in samples.iterrows():
            result = self.process_sample(sample)
            if result is not None:
                results.append(result)

        return results

    def flush_window(self) -> Optional[WindowProcessingResult]:
        """Force processing of the current window even if not full.

        Useful at the end of a stream to process any remaining samples.

        Returns:
            WindowProcessingResult if buffer contains samples, None if empty

        Raises:
            ValueError: If model is not trained and initial_training_required is True
        """
        self._check_if_trained()

        if not self._buffer.has_samples():
            return None

        return self._process_window()

    def _process_window(self) -> WindowProcessingResult:
        """Core window processing logic.

        Steps:
        1. Convert buffer to DataFrame
        2. Score all samples with EOTE
        3. Calculate outlier ratio (for statistics)
        4. Detect distribution drift using KS test
        5. If drift detected, retrain on bottom N% of scores and re-score
        6. Update statistics and clear buffer

        Returns:
            WindowProcessingResult containing window statistics and sample results
        """
        window_id = self._buffer.current_window_id
        window_df = self._buffer.get_dataframe()
        window_size = len(window_df)

        # Score all samples
        scores = self._eote.classify(window_df)  # Returns List[List[float]]

        # Calculate outlier ratio (for statistics only)
        outlier_ratio = self._calculate_outlier_ratio(scores)
        outlier_count = int(outlier_ratio * window_size)

        # Detect distribution drift
        drift_detected, p_value, ks_statistic = self._detect_distribution_drift(scores)

        self._logger.log_window_processed(window_id, outlier_ratio)

        # Check if retraining should be triggered
        retraining_triggered = False
        retraining_skipped = False
        training_samples_used = 0
        original_scores = None

        if drift_detected:
            # Filter samples using adaptive percentile
            training_df = self._filter_samples_for_retraining(window_df, scores)
            training_count = len(training_df)

            if training_count >= self._config.min_normal_samples:
                # Retrain on filtered samples
                try:
                    self._logger.log_retraining_triggered(
                        window_id, training_count, p_value, ks_statistic
                    )
                    original_scores = scores
                    labels = self._create_normal_labels(training_count)
                    self._eote.train(training_df, labels)
                    scores = self._eote.classify(window_df)  # Re-score with new model
                    retraining_triggered = True
                    training_samples_used = training_count

                    # Update baseline scores after retraining to reflect new model
                    window_scores = np.array([s[0] for s in scores])
                    self._baseline_scores = window_scores.copy()

                    self._logger.log_retraining_completed(window_id)
                except Exception as e:
                    self._logger.log_error(
                        f"Retraining failed for window {window_id}: {str(e)}"
                    )
                    retraining_skipped = True
                    # Keep original scores and old model
            else:
                self._logger.log_retraining_skipped(
                    window_id,
                    f"insufficient samples ({training_count} < {self._config.min_normal_samples})"
                )
                retraining_skipped = True

        # Create sample results
        sample_results = []
        for i, score_list in enumerate(scores):
            score = score_list[0]  # EOTE returns List[List[float]]
            is_outlier = score > self._config.outlier_threshold
            was_rescored = retraining_triggered
            orig_score = original_scores[i][0] if original_scores else None

            sample_results.append(StreamingSampleResult(
                sample_index=i,
                score=score,
                is_outlier=is_outlier,
                window_id=window_id,
                was_rescored=was_rescored,
                original_score=orig_score
            ))

        # Update statistics
        self._statistics.record_window_processed(
            sample_count=window_size,
            outlier_count=outlier_count,
            retraining_triggered=retraining_triggered,
            retraining_skipped=retraining_skipped,
            drift_detected=drift_detected
        )

        # Clear buffer for next window
        self._buffer.clear()

        return WindowProcessingResult(
            window_id=window_id,
            total_samples=window_size,
            outlier_count=outlier_count,
            outlier_ratio=outlier_ratio,
            retraining_triggered=retraining_triggered,
            training_samples_used=training_samples_used,
            samples_results=sample_results,
            drift_detected=drift_detected,
            drift_p_value=p_value,
            drift_ks_statistic=ks_statistic
        )

    def _calculate_outlier_ratio(self, scores: List[List[float]]) -> float:
        """Calculate the proportion of scores exceeding the outlier threshold.

        Args:
            scores: List of score lists from EOTE.classify()

        Returns:
            Proportion of outliers (0.0 to 1.0)
        """
        if not scores:
            return 0.0

        outlier_count = sum(1 for score_list in scores
                          if score_list[0] > self._config.outlier_threshold)
        return outlier_count / len(scores)

    def _detect_distribution_drift(self, scores: List[List[float]]) -> tuple:
        """Detect distribution drift using Kolmogorov-Smirnov test.

        Compares current window scores to baseline scores using
        the KS two-sample test. Drift is detected when p-value falls below
        the configured significance level.

        For the first window after training, baseline scores are updated
        from the window (not from training data to avoid feature encoding issues).

        Args:
            scores: Current window scores from EOTE.classify()

        Returns:
            Tuple of (drift_detected: bool, p_value: float, ks_statistic: float)
        """
        window_scores = np.array([s[0] for s in scores])

        # First window or if baseline failed: establish baseline and skip drift detection
        if self._baseline_scores is None or len(self._baseline_scores) == 0:
            self._baseline_scores = window_scores.copy()
            return False, 1.0, 0.0

        # Subsequent windows: perform KS test
        statistic, p_value = ks_2samp(self._baseline_scores, window_scores)
        drift_detected = p_value < self._config.drift_significance_level

        return drift_detected, p_value, statistic

    def _filter_samples_for_retraining(
        self,
        window_df: pd.DataFrame,
        scores: List[List[float]]
    ) -> pd.DataFrame:
        """Filter samples using adaptive percentile threshold.

        Keeps the bottom N% of scores (by default 75%) for retraining,
        excluding only extreme outliers. This ensures sufficient samples
        for retraining even during concept drift.

        Args:
            window_df: DataFrame containing all window samples
            scores: Anomaly scores for each sample

        Returns:
            DataFrame containing samples for retraining (bottom N percentile)
        """
        score_values = np.array([s[0] for s in scores])
        percentile_threshold = np.percentile(
            score_values,
            self._config.retraining_percentile * 100
        )

        training_mask = [s[0] <= percentile_threshold for s in scores]
        return window_df[training_mask].reset_index(drop=True)

    def _create_normal_labels(self, count: int) -> pd.DataFrame:
        """Create a DataFrame of 'normal' labels for retraining.

        EOTE requires a target DataFrame (single column) for training. In streaming mode, all
        samples used for retraining are assumed to be normal.

        Args:
            count: Number of labels to generate

        Returns:
            DataFrame with single column containing 'normal' repeated count times
        """
        return pd.DataFrame({'class': ['normal'] * count})

    def _check_if_trained(self) -> None:
        """Verify model is trained before processing.

        Raises:
            ValueError: If initial_training_required is True and model is not trained
        """
        if self._config.initial_training_required and not self._is_trained:
            raise ValueError(
                "Model must be trained before processing samples. "
                "Call train() first or set initial_training_required=False."
            )

    def get_statistics(self) -> StreamingStatistics:
        """Get current streaming statistics.

        Returns:
            StreamingStatistics object with cumulative metrics
        """
        return self._statistics

    def is_trained(self) -> bool:
        """Check if the model has been trained.

        Returns:
            True if train() has been called successfully
        """
        return self._is_trained

    def has_pending_samples(self) -> bool:
        """Check if there are samples in the buffer waiting to be processed.

        Returns:
            True if buffer contains unprocessed samples
        """
        return self._buffer.has_samples()

    def reset(self, keep_model: bool = False) -> None:
        """Reset streaming state.

        Args:
            keep_model: If True, retains the trained model. If False, clears the model
                       and requires retraining before processing more samples.
        """
        self._buffer = WindowBuffer(self._config.window_size)
        self._statistics.reset()

        if not keep_model:
            # Create new EOTE instance
            director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
            self._eote = director.get_eote()
            self._is_trained = False
            self._logger.log_info("Streaming state and model reset")
        else:
            self._logger.log_info("Streaming state reset (model retained)")
