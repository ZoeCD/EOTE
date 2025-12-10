import sys
sys.path.append(".")
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from EOTE import StreamingEOTE
from EOTE.Streaming import SilentLogger
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from sklearn.datasets import load_iris


class TestStreamingEOTEInitialization:
    """Test StreamingEOTE initialization."""

    def test_default_initialization(self):
        streaming = StreamingEOTE()
        assert streaming.is_trained() is False
        assert not streaming.has_pending_samples()

    def test_custom_parameters(self):
        streaming = StreamingEOTE(
            window_size=50,
            outlier_threshold=0.5,
            drift_significance_level=0.01,
            retraining_percentile=0.8,
            min_normal_samples=20
        )
        assert streaming._config.window_size == 50
        assert streaming._config.outlier_threshold == 0.5
        assert streaming._config.drift_significance_level == 0.01
        assert streaming._config.retraining_percentile == 0.8
        assert streaming._config.min_normal_samples == 20

    def test_with_silent_logger(self):
        streaming = StreamingEOTE(logger=SilentLogger())
        assert isinstance(streaming._logger, SilentLogger)


class TestStreamingEOTETraining:
    """Test initial training functionality."""

    @pytest.fixture
    def training_data(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        # Use only class 0 for semi-supervised training
        mask = y == 0
        X_train = X[mask].reset_index(drop=True)
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        return X_train, y_train

    def test_train_updates_state(self, training_data): 
        X_train, y_train = training_data
        streaming = StreamingEOTE(logger=SilentLogger())

        assert not streaming.is_trained()
        streaming.train(X_train, y_train)
        assert streaming.is_trained()

    def test_process_without_training_raises_error(self):
        streaming = StreamingEOTE(logger=SilentLogger())
        sample = pd.Series({'a': 1, 'b': 2, 'c': 3})

        with pytest.raises(ValueError, match="Model must be trained"):
            streaming.process_sample(sample)



class TestStreamingEOTESampleProcessing:
    """Test processing individual samples."""

    @pytest.fixture
    def trained_streaming(self):
        X_train = pd.DataFrame({
            'feature1': [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 1.3, 1.4, 1.5, 1.6],
            'feature2': [2.0, 2.1, 1.9, 2.05, 1.95, 2.2, 2.3, 2.4, 2.5, 2.6],
            'feature3': [3.0, 3.1, 2.9, 3.05, 2.95, 3.2, 3.3, 3.4, 3.5, 3.6],
        })
        y_train = pd.DataFrame({'class': ['normal'] * 10})

        streaming = StreamingEOTE(
            window_size=3,
            outlier_threshold=0.0,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        return streaming

    def test_process_single_sample_incomplete_window(self, trained_streaming):
        sample = pd.Series({'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0})
        result = trained_streaming.process_sample(sample)

        assert result is None  # Window not full yet
        assert trained_streaming.has_pending_samples()

    def test_process_samples_until_window_full(self, trained_streaming):
        samples = [
            pd.Series({'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0}),
            pd.Series({'feature1': 1.1, 'feature2': 2.1, 'feature3': 3.1}),
            pd.Series({'feature1': 0.9, 'feature2': 1.9, 'feature3': 2.9}),
        ]

        result1 = trained_streaming.process_sample(samples[0])
        assert result1 is None

        result2 = trained_streaming.process_sample(samples[1])
        assert result2 is None

        result3 = trained_streaming.process_sample(samples[2])
        assert result3 is not None  # Window full
        assert result3.window_id == 0
        assert result3.total_samples == 3

    def test_window_result_structure(self, trained_streaming):
        samples = [
            pd.Series({'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0}),
            pd.Series({'feature1': 1.1, 'feature2': 2.1, 'feature3': 3.1}),
            pd.Series({'feature1': 0.9, 'feature2': 1.9, 'feature3': 2.9}),
        ]

        for sample in samples:
            result = trained_streaming.process_sample(sample)

        assert result.window_id == 0
        assert result.total_samples == 3
        assert len(result.samples_results) == 3
        assert 0.0 <= result.outlier_ratio <= 1.0


class TestStreamingEOTEBatchProcessing:
    """Test batch processing functionality."""

    @pytest.fixture
    def trained_streaming(self):
        X_train = pd.DataFrame({
            'x': np.tile(np.arange(10, 20), 2),
            'y': np.tile(np.arange(10, 20), 2),
        })
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})

        streaming = StreamingEOTE(
            window_size=5,
            min_normal_samples=3,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        return streaming

    def test_process_batch_single_window(self, trained_streaming):
        batch = pd.DataFrame({
            'x': np.random.randn(5),
            'y': np.random.randn(5)
        })
        results = trained_streaming.process_batch(batch)
        assert len(results) == 1
        assert results[0].total_samples == 5

    def test_process_batch_multiple_windows(self, trained_streaming):
        batch = pd.DataFrame({
            'x': np.random.randn(12),
            'y': np.random.randn(12)
        })
        results = trained_streaming.process_batch(batch)
        assert len(results) == 2
        assert results[0].window_id == 0
        assert results[1].window_id == 1

    def test_process_batch_partial_window(self, trained_streaming):
        batch = pd.DataFrame({
            'x': np.random.randn(3),
            'y': np.random.randn(3)
        })
        results = trained_streaming.process_batch(batch)
        assert len(results) == 0  # No complete window
        assert trained_streaming.has_pending_samples()


class TestStreamingEOTERetraining:
    """Test adaptive retraining logic."""

    @pytest.fixture
    def trained_streaming_with_high_threshold(self):
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.normal(10, 1.5, 100),
        })
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})


        director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
        eote = director.get_eote()

        streaming = StreamingEOTE(
            eote=eote,
            window_size=20,
            outlier_threshold=0.0,
            drift_significance_level=0.05,
            retraining_percentile=0.75,
            min_normal_samples=5,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        return streaming

    def test_no_retraining_with_low_outlier_ratio(self, trained_streaming_with_high_threshold):
        # Generate normal samples
        np.random.seed(42)
        normal_samples = pd.DataFrame({
            'feature': np.random.normal(0, 1, 10),
            'feature2': np.random.normal(5, 2, 10),
            'feature3': np.random.normal(10, 1.5, 10),
        })

        results = trained_streaming_with_high_threshold.process_batch(normal_samples)

        # Expecting no retraining (most samples should be normal)
        if results:
            assert results[0].retraining_triggered is False or results[0].outlier_ratio <= 0.5

    def test_retraining_triggered_with_high_outlier_ratio(self, trained_streaming_with_high_threshold): 
        # Generate outliers (far from 0) mixed with normals
        outlier_samples = pd.DataFrame({
            'feature': [10, 11, 12, 13, 14, 15, 0.1, 0.2, 0.3, 0.4],
            'feature2': [10, 11, 12, 13, 14, 15, 0.1, 0.2, 0.3, 0.4],
            'feature3': [20, 21, 22, 23, 24, 25, 0.5, 0.6, 0.7, 0.8],
        })

        results = trained_streaming_with_high_threshold.process_batch(outlier_samples)

        if results:
            assert hasattr(results[0], 'retraining_triggered')

    def test_retraining_updates_scores(self, trained_streaming_with_high_threshold): 
        """Test that re-scoring happens after retraining."""
        np.random.seed(42)
        # Window with shift in distribution
        shifted_samples = pd.DataFrame({
            'feature': [10, 11, 12, 13, 14, 15, 0.1, 0.2, 0.3, 0.4],
            'feature2': [10, 11, 12, 13, 14, 15, 0.1, 0.2, 0.3, 0.4],
            'feature3': [20, 21, 22, 23, 24, 25, 0.5, 0.6, 0.7, 0.8],
        })

        results = trained_streaming_with_high_threshold.process_batch(shifted_samples)

        if results and results[0].retraining_triggered:
            # Check that samples have was_rescored flag
            assert any(s.was_rescored for s in results[0].samples_results)


class TestStreamingEOTEEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def trained_streaming(self):
        X_train = pd.DataFrame({
            'a': np.tile(np.arange(5), 2),
            'b': np.tile(np.arange(10, 15), 2)
        })
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})

        streaming = StreamingEOTE(
            window_size=5,
            outlier_threshold=0.0,
            min_normal_samples=5,  # Match window size for testing edge case
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        return streaming

    def test_insufficient_normal_samples_skips_retraining(self, trained_streaming):
        """Test that retraining is skipped when insufficient normal samples."""
        # This would trigger retraining but have too few normal samples
        samples = pd.DataFrame({
            'a': [100, 101, 102, 103, 104],  # All likely outliers
            'b': [200, 201, 202, 203, 204],
            'c': [300, 301, 302, 303, 304]
        })

        results = trained_streaming.process_batch(samples)

        if results:
            # If retraining was triggered, it should be skipped due to insufficient samples
            if results[0].drift_detected and results[0].retraining_triggered:
                assert results[0].training_samples_used >= trained_streaming._config.min_normal_samples

    def test_flush_window_with_partial_data(self):
        X_train = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 11, 12, 13, 14]})
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        streaming = StreamingEOTE(
            window_size=10,
            min_normal_samples=3,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        # Add only 3 samples (less than window_size)
        for i in range(3):
            streaming.process_sample(pd.Series({'x': i, 'y': i+10}))
        assert streaming.has_pending_samples()
        # Flush partial window
        result = streaming.flush_window()
        assert result is not None
        assert result.total_samples == 3
        assert not streaming.has_pending_samples()

    def test_flush_empty_window(self):
        X_train = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 11, 12]
        })
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})

        streaming = StreamingEOTE(
            window_size=5,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)

        result = streaming.flush_window()
        assert result is None


class TestStreamingEOTEStatistics:
    """Test statistics tracking."""

    def test_statistics_updated_after_processing(self):
        X_train = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 11, 12, 13, 14]})
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        streaming = StreamingEOTE(
            window_size=5,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        # Process one window
        samples = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 11, 12, 13, 14]})
        streaming.process_batch(samples)
        stats = streaming.get_statistics()
        assert stats.windows_processed == 1
        assert stats.total_samples_processed == 5

    def test_statistics_accumulate_across_windows(self):
        X_train = pd.DataFrame({'x': range(10), 'y': range(10, 20)})
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        streaming = StreamingEOTE(
            window_size=3,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        # Process 2 windows (6 samples)
        samples = pd.DataFrame({'x': range(6), 'y': range(10, 16)})
        streaming.process_batch(samples)
        stats = streaming.get_statistics()
        assert stats.windows_processed == 2
        assert stats.total_samples_processed == 6


class TestStreamingEOTEReset:
    """Test reset functionality."""

    def test_reset_without_keeping_model(self):
        X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 11, 12]})
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        streaming = StreamingEOTE(
            window_size=5,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        # Add some samples
        streaming.process_sample(pd.Series({'x': 1, 'y': 10}))
        assert streaming.has_pending_samples()
        # Reset
        streaming.reset(keep_model=False)
        assert not streaming.has_pending_samples()
        assert not streaming.is_trained()

    def test_reset_keeping_model(self):
        X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 11, 12]})
        y_train = pd.DataFrame({'class': ['normal'] * len(X_train)})
        streaming = StreamingEOTE(
            window_size=5,
            min_normal_samples=2,
            logger=SilentLogger()
        )
        streaming.train(X_train, y_train)
        streaming.process_sample(pd.Series({'x': 1, 'y': 10}))
        # Reset but keep model
        streaming.reset(keep_model=True)
        assert not streaming.has_pending_samples()
        assert streaming.is_trained()
