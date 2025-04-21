from typing import Protocol
from pandas import DataFrame


class DataFrameBuilder(Protocol):
    def create_dataframe(self, file_path: str) -> DataFrame:
        ...
