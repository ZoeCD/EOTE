import numpy as np
from typing import Optional, List


class Feature:
    def __init__(self, name: str, values: Optional[List] = None):
        self._name = name
        self._values = values
        self._missing_value_objects = []
        self._type = self._set_type(values)

    def name(self) -> str:
        return self._name

    def values(self) -> List:
        return self._values

    def type(self) -> str:
        return self._type

    def set_values(self, values: List) -> None:
        self._values = values
        self._type = self._set_type(values)

    def set_missing_value_objects(self, indexes: List) -> None:
        self._missing_value_objects = indexes

    def _set_type(self, values: Optional[np.ndarray]) -> str:
        if values is None or len(values) == 0:
            return None
        return 'Numerical' if issubclass(values.dtype.type, (np.integer, np.floating)) else 'Nominal'

    @property
    def missing_value_objects(self) -> List:
        return self._missing_value_objects


