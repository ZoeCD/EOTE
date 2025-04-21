from typing import Protocol, Union
from pandas import DataFrame, Series


class AttributeRemover(Protocol):
    def remove_invalid_attributes(self, dataset: DataFrame) -> DataFrame:
        ...


class DataVerifier(Protocol):
    def verify_if_valid(self, data: Union[DataFrame, Series]) -> bool:
        ...

    def raise_exception(self):
        ...

