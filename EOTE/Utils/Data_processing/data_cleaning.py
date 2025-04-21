from EOTE.Protocols import AttributeRemover
from typing import Union
from pandas.api.types import is_string_dtype, is_object_dtype
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from EOTE.Protocols import DataVerifier



class InsufficientCategoricalValuesAttributeRemover(AttributeRemover):
    def __init__(self, min_values: int = None, min_instances: int = None):
        self.min_values = min_values
        self.min_instances = min_instances

    def remove_invalid_attributes(self, dataset: DataFrame) -> DataFrame:
        return self.__clean_attributes(dataset)

    def __clean_attributes(self, dataset: DataFrame) -> DataFrame:
        invalid_attributes = list()
        for i in range(len(dataset.columns)):
            attribute = dataset.columns[i]
            if is_string_dtype(dataset[attribute]) or is_object_dtype(dataset[attribute]):
                if not self.__verify_attribute(attribute, dataset):
                    invalid_attributes.append(attribute)
        return dataset.drop(columns=invalid_attributes)

    def __verify_attribute(self, attribute: str, dataset: DataFrame) -> bool:
        return self.verify_if_valid(dataset[attribute])

    def verify_if_valid(self, data: Union[DataFrame, Series]) -> bool:
        return True if (np.sum(data.value_counts() >= self.min_instances) >= self.min_values) else False


class UniqueClassVerifier(DataVerifier):
    def verify_if_valid(self, data: Union[DataFrame, Series]) -> bool:
        try:
            if data.iloc[:, 0].nunique() == 1:
                return True
            else:
                self.raise_exception()
        except:
            raise ValueError("Invalid Class: No objects in dataframe!")

    def raise_exception(self):
        raise ValueError(
            f"Invalid Class: The dataset must contain objects of a single class!")

