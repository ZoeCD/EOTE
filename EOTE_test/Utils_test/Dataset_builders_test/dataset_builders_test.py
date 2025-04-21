import sys
sys.path.append(".")
import unittest
import pandas as pd
from EOTE.Utils import DataFrameBuilderAff


class TestDataFrameBuilders(unittest.TestCase):

    def test_arff_to_dataframe_categorical(self):
        expected_dataframe = pd.read_csv(
            "EOTE_test/Datasets/datasets_csv/categorical_training.csv")

        dataframe_builder = DataFrameBuilderAff()
        actual_dataframe = dataframe_builder.create_dataframe(
            "EOTE_test/Datasets/datasets_arff/categorical_training.arff")
        pd.testing.assert_frame_equal(expected_dataframe, actual_dataframe)

    def test_arff_to_dataframe_numerical(self):
        expected_dataframe = pd.read_csv(
            "EOTE_test/Datasets/datasets_csv/numerical_training.csv")

        dataframe_builder = DataFrameBuilderAff()
        actual_dataframe = dataframe_builder.create_dataframe(
            "EOTE_test/Datasets/datasets_arff/numerical_training.arff")
        pd.testing.assert_frame_equal(expected_dataframe, actual_dataframe)


    def test_arff_to_dataframe_mixed(self):
        expected_dataframe = pd.read_csv(
            "EOTE_test/Datasets/datasets_csv/mixed_training.csv")

        dataframe_builder = DataFrameBuilderAff()
        actual_dataframe = dataframe_builder.create_dataframe(
            "EOTE_test/Datasets/datasets_arff/mixed_training.arff")
        pd.testing.assert_frame_equal(expected_dataframe, actual_dataframe)

