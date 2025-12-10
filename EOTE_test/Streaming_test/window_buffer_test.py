import sys
sys.path.append(".")
import pytest
import pandas as pd
import numpy as np
from EOTE.Streaming.window_buffer import WindowBuffer


class TestWindowBufferInitialization:
    """Test WindowBuffer initialization and validation."""

    def test_valid_initialization(self):
        buffer = WindowBuffer(window_size=10)
        assert buffer.window_size == 10
        assert buffer.current_window_id == 0
        assert buffer.size() == 0

    def test_invalid_window_size_zero(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowBuffer(window_size=0)

    def test_invalid_window_size_negative(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            WindowBuffer(window_size=-5)


class TestWindowBufferSampleAccumulation:
    """Test adding samples and buffer filling."""

    @pytest.fixture
    def buffer(self):
        return WindowBuffer(window_size=3)

    @pytest.fixture
    def sample_data(self):
        return [
            pd.Series({'age': 25, 'income': 50000, 'city': 'NYC'}, name=0),
            pd.Series({'age': 30, 'income': 60000, 'city': 'LA'}, name=1),
            pd.Series({'age': 35, 'income': 70000, 'city': 'SF'}, name=2),
            pd.Series({'age': 40, 'income': 80000, 'city': 'Boston'}, name=3),
        ]

    def test_add_single_sample(self, buffer, sample_data):
        is_full = buffer.add_sample(sample_data[0])
        assert not is_full
        assert buffer.size() == 1
        assert buffer.has_samples()

    def test_buffer_not_full_before_capacity(self, buffer, sample_data):
        buffer.add_sample(sample_data[0])
        buffer.add_sample(sample_data[1])
        assert buffer.size() == 2
        assert not buffer.is_full()

    def test_buffer_full_at_exact_capacity(self, buffer, sample_data):
        buffer.add_sample(sample_data[0])
        buffer.add_sample(sample_data[1])
        is_full = buffer.add_sample(sample_data[2])
        assert is_full
        assert buffer.is_full()
        assert buffer.size() == 3

    def test_buffer_full_beyond_capacity(self, buffer, sample_data):
        buffer.add_sample(sample_data[0])
        buffer.add_sample(sample_data[1])
        buffer.add_sample(sample_data[2])

        with pytest.raises(BufferError, match="Buffer is already full; cannot add more samples."):
            buffer.add_sample(sample_data[3])
        assert buffer.size() == 3


class TestWindowBufferDataFrameConversion:
    """Test converting buffer to DataFrame."""

    @pytest.fixture
    def buffer(self):
        return WindowBuffer(window_size=3)

    def test_empty_buffer_to_dataframe(self, buffer):
        df = buffer.get_dataframe()
        assert df.empty
        assert isinstance(df, pd.DataFrame)

    def test_single_sample_to_dataframe(self, buffer):
        sample = pd.Series({'a': 1, 'b': 2, 'c': 3})
        buffer.add_sample(sample)
        df = buffer.get_dataframe()

        assert len(df) == 1
        assert list(df.columns) == ['a', 'b', 'c']
        assert df.iloc[0]['a'] == 1
        assert df.iloc[0]['b'] == 2
        assert df.iloc[0]['c'] == 3

    def test_multiple_samples_to_dataframe(self, buffer):
        samples = [
            pd.Series({'x': 10, 'y': 20}),
            pd.Series({'x': 30, 'y': 40}),
            pd.Series({'x': 50, 'y': 60}),
        ]

        for sample in samples:
            buffer.add_sample(sample)

        df = buffer.get_dataframe()

        assert len(df) == 3
        assert list(df.columns) == ['x', 'y']
        assert df.iloc[0]['x'] == 10
        assert df.iloc[1]['x'] == 30
        assert df.iloc[2]['x'] == 50

    def test_dataframe_index_reset(self, buffer):
        samples = [
            pd.Series({'val': i}, name=i*10) for i in range(3)
        ]

        for sample in samples:
            buffer.add_sample(sample)

        df = buffer.get_dataframe()
        assert list(df.index) == [0, 1, 2]

    def test_dataframe_preserves_data_types(self, buffer):
        sample = pd.Series({
            'int_col': 42,
            'float_col': 3.14,
            'str_col': 'hello'
        })
        buffer.add_sample(sample)
        df = buffer.get_dataframe()

        assert df.iloc[0]['int_col'] == 42
        assert df.iloc[0]['float_col'] == 3.14
        assert df.iloc[0]['str_col'] == 'hello'

    def test_sample_copy_independence(self, buffer):
        """Ensure added samples are copied and don't affect buffer if modified."""
        original = pd.Series({'a': 1, 'b': 2})
        buffer.add_sample(original)

        # Modify original
        original['a'] = 999

        # Buffer should still have original value
        df = buffer.get_dataframe()
        assert df.iloc[0]['a'] == 1


class TestWindowBufferClearAndReset:
    """Test buffer clearing and window ID management."""

    @pytest.fixture
    def buffer(self):
        return WindowBuffer(window_size=5)

    def test_clear_empty_buffer(self, buffer):
        initial_id = buffer.current_window_id
        buffer.clear()
        assert buffer.size() == 0
        assert buffer.current_window_id == initial_id + 1

    def test_clear_with_samples(self, buffer):
        samples = [pd.Series({'x': i}) for i in range(3)]
        for sample in samples:
            buffer.add_sample(sample)

        assert buffer.size() == 3
        buffer.clear()

        assert buffer.size() == 0
        assert not buffer.has_samples()

    def test_window_id_increments_on_clear(self, buffer):
        assert buffer.current_window_id == 0

        buffer.clear()
        assert buffer.current_window_id == 1

        buffer.clear()
        assert buffer.current_window_id == 2

    def test_reuse_after_clear(self, buffer):
        # Fill and clear
        samples_1 = [pd.Series({'x': i}) for i in range(3)]
        for sample in samples_1:
            buffer.add_sample(sample)
        buffer.clear()

        # Reuse buffer
        samples_2 = [pd.Series({'y': i*10}) for i in range(2)]
        for sample in samples_2:
            buffer.add_sample(sample)

        assert buffer.size() == 2
        df = buffer.get_dataframe()
        assert list(df.columns) == ['y']
        assert df.iloc[0]['y'] == 0


class TestWindowBufferEdgeCases:
    """Test edge cases and special scenarios."""

    def test_window_size_one(self):
        buffer = WindowBuffer(window_size=1)
        sample = pd.Series({'a': 1})

        is_full = buffer.add_sample(sample)
        assert is_full
        assert buffer.size() == 1

    def test_large_window_size(self):
        buffer = WindowBuffer(window_size=10000)
        assert buffer.window_size == 10000
        assert not buffer.is_full()

    def test_has_samples_empty(self):
        buffer = WindowBuffer(window_size=5)
        assert not buffer.has_samples()

    def test_has_samples_with_data(self):
        buffer = WindowBuffer(window_size=5)
        buffer.add_sample(pd.Series({'x': 1}))
        assert buffer.has_samples()

    def test_has_samples_after_clear(self):
        buffer = WindowBuffer(window_size=5)
        buffer.add_sample(pd.Series({'x': 1}))
        buffer.clear()
        assert not buffer.has_samples()


class TestWindowBufferIntegrationScenarios:
    """Test realistic streaming scenarios."""

    def test_continuous_streaming_multiple_windows(self):
        buffer = WindowBuffer(window_size=5)
        window_results = []

        # Simulate 12 samples (2.4 windows)
        for i in range(12):
            sample = pd.Series({'value': i, 'squared': i**2})
            is_full = buffer.add_sample(sample)

            if is_full:
                df = buffer.get_dataframe()
                window_results.append({
                    'window_id': buffer.current_window_id,
                    'size': len(df),
                    'data': df.copy()
                })
                buffer.clear()

        # Should have processed 2 complete windows
        assert len(window_results) == 2

        # Window 0
        assert window_results[0]['window_id'] == 0
        assert window_results[0]['size'] == 5
        assert list(window_results[0]['data']['value']) == [0, 1, 2, 3, 4]

        # Window 1
        assert window_results[1]['window_id'] == 1
        assert window_results[1]['size'] == 5
        assert list(window_results[1]['data']['value']) == [5, 6, 7, 8, 9]

        # Remaining 2 samples in buffer
        assert buffer.size() == 2
        assert buffer.current_window_id == 2
