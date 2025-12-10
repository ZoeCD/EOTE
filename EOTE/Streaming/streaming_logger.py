from typing import Protocol


class StreamingLogger(Protocol):
    """Protocol for logging streaming EOTE events.

    Implementations can customize logging behavior (e.g., file output, external
    logging systems, structured logging). The default implementation logs to stdout.
    """

    def log_window_processed(self, window_id: int, outlier_ratio: float) -> None:
        """Log completion of window processing.

        Args:
            window_id: ID of the processed window
            outlier_ratio: Proportion of outliers in the window (0.0 to 1.0)
        """
        ...

    def log_retraining_triggered(
        self,
        window_id: int,
        sample_count: int,
        p_value: float = None,
        ks_statistic: float = None
    ) -> None:
        """Log that model retraining has been triggered.

        Args:
            window_id: ID of the window triggering retraining
            sample_count: Number of samples to be used for retraining
            p_value: KS test p-value (optional)
            ks_statistic: KS test statistic (optional)
        """
        ...

    def log_retraining_completed(self, window_id: int) -> None:
        """Log successful completion of model retraining.

        Args:
            window_id: ID of the window that triggered retraining
        """
        ...

    def log_retraining_skipped(self, window_id: int, reason: str) -> None:
        """Log that retraining was triggered but skipped.

        Args:
            window_id: ID of the window that would have triggered retraining
            reason: Explanation for why retraining was skipped
        """
        ...

    def log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning message text
        """
        ...

    def log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Error message text
        """
        ...

    def log_info(self, message: str) -> None:
        """Log an informational message.

        Args:
            message: Info message text
        """
        ...


class DefaultStreamingLogger:
    """Default logger implementation that outputs to stdout.

    Simple console logger with prefix "[EOTE-Stream]" for easy identification
    of streaming-related log messages.

    Attributes:
        verbose: If True, logs all events. If False, only logs warnings and errors.
    """

    def __init__(self, verbose: bool = True):
        """Initialize logger.

        Args:
            verbose: Enable verbose logging (info, window processing, retraining events)
        """
        self.verbose = verbose

    def log_window_processed(self, window_id: int, outlier_ratio: float) -> None:
        if self.verbose:
            print(f"[EOTE-Stream] Window {window_id} processed: "
                  f"{outlier_ratio:.2%} outliers")

    def log_retraining_triggered(
        self,
        window_id: int,
        sample_count: int,
        p_value: float = None,
        ks_statistic: float = None
    ) -> None:
        msg = f"[EOTE-Stream] Window {window_id}: Retraining triggered with {sample_count} samples"
        if p_value is not None:
            msg += f" (KS p-value={p_value:.4f}, statistic={ks_statistic:.4f})"
        print(msg)

    def log_retraining_completed(self, window_id: int) -> None:
        print(f"[EOTE-Stream] Window {window_id}: Retraining completed successfully")

    def log_retraining_skipped(self, window_id: int, reason: str) -> None:
        print(f"[EOTE-Stream] Window {window_id}: Retraining skipped - {reason}")

    def log_warning(self, message: str) -> None:
        print(f"[EOTE-Stream] WARNING: {message}")

    def log_error(self, message: str) -> None:
        print(f"[EOTE-Stream] ERROR: {message}")

    def log_info(self, message: str) -> None:
        if self.verbose:
            print(f"[EOTE-Stream] {message}")


class SilentLogger:
    """Logger that suppresses all output.

    Useful for testing or when logging is not desired.
    """

    def log_window_processed(self, window_id: int, outlier_ratio: float) -> None:
        pass

    def log_retraining_triggered(
        self,
        window_id: int,
        sample_count: int,
        p_value: float = None,
        ks_statistic: float = None
    ) -> None:
        pass

    def log_retraining_completed(self, window_id: int) -> None:
        pass

    def log_retraining_skipped(self, window_id: int, reason: str) -> None:
        pass

    def log_warning(self, message: str) -> None:
        pass

    def log_error(self, message: str) -> None:
        pass

    def log_info(self, message: str) -> None:
        pass
