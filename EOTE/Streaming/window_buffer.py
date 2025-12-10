import pandas as pd
from typing import List


class WindowBuffer:
    """Buffer for accumulating samples into fixed-size windows.

    The buffer collects pandas Series samples and converts them to a DataFrame
    when the window is full for batch processing by EOTE.

    Attributes:
        window_size: Maximum number of samples per window
        current_window_id: ID of the current window being accumulated
    """

    def __init__(self, window_size: int):
        """Initialize window buffer.

        Args:
            window_size: Number of samples per window (must be positive)

        Raises:
            ValueError: If window_size is not positive
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.window_size = window_size
        self._samples: List[pd.Series] = []
        self.current_window_id: int = 0

    def add_sample(self, sample: pd.Series) -> bool:
        """Add a sample to the buffer.

        Args:
            sample: pandas Series representing a single sample

        Raises:
            BufferError: If trying to add a sample when buffer is already full

        Returns:
            True if the window is now full (size >= window_size), False otherwise
        """
        if self.is_full():
            raise BufferError("Buffer is already full; cannot add more samples.")
        self._samples.append(sample.copy())
        return self.is_full()

    def is_full(self) -> bool:
        """Check if the window has reached capacity.

        Returns:
            True if buffer contains window_size or more samples
        """
        return len(self._samples) >= self.window_size

    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffered samples to a DataFrame.

        Returns:
            DataFrame with samples as rows, features as columns.
            Returns empty DataFrame if buffer is empty.

        Note:
            Index is reset to 0-based integer range for consistent processing.
        """
        if not self._samples:
            return pd.DataFrame()

        # Convert list of Series to DataFrame (each Series becomes a row)
        df = pd.concat(self._samples, axis=1).T.reset_index(drop=True)
        return df

    def clear(self) -> None:
        """Clear all buffered samples and increment window ID.

        Use this after processing a complete window.
        """
        self._samples = []
        self.current_window_id += 1

    def size(self) -> int:
        """Get the current number of samples in the buffer.

        Returns:
            Number of samples currently buffered
        """
        return len(self._samples)

    def has_samples(self) -> bool:
        """Check if buffer contains any samples.

        Returns:
            True if buffer is not empty
        """
        return len(self._samples) > 0
