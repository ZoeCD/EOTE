from pandas import DataFrame
from typing import Protocol


class ImputerInterface(Protocol):

    def fit_transform(self, data: DataFrame) -> DataFrame:
        ...

    def fit(self, data: DataFrame) -> None:
        ...

    def transform(self, data: DataFrame) -> DataFrame:
        ...
