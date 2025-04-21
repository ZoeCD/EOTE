import arff
from EOTE.Protocols import DataFrameBuilder
import pandas as pd


class DataFrameBuilderAff(DataFrameBuilder):

    def create_dataframe(self, file_path: str) -> pd.DataFrame:
        file = open(file_path, "r")
        dataset = arff.load(file)
        file.close()
        feature_names = list()
        for attribute in dataset['attributes']:
            feature_names.append(attribute[0])
        dataset = pd.DataFrame(dataset['data'], columns=feature_names)
        return dataset

