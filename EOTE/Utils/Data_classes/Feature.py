import numpy as np
from typing import Optional, List


class Feature:
    """Encapsulates metadata about a dataset feature.

    Stores information about a feature including its name, unique values, type
    (Numerical or Nominal), and which instances have missing values for this feature.

    The feature type is automatically detected based on the numpy dtype of the values:
    - Numerical: integer or floating point types
    - Nominal: all other types (strings, objects, etc.)

    Attributes:
        _name: Feature name as it appears in the dataset
        _values: Array of unique values observed for this feature
        _missing_value_objects: List of row indices where this feature is missing
        _type: Either 'Numerical' or 'Nominal'

    Example:
        >>> feature = Feature(name="age", values=np.array([25, 30, 35]))
        >>> feature.type()
        'Numerical'
        >>> feature.set_missing_value_objects([2, 5, 7])
        >>> feature.missing_value_objects
        [2, 5, 7]
    """

    def __init__(self, name: str, values: Optional[List] = None):
        """Initialize a Feature with name and optional values.

        Args:
            name: Name of the feature as it appears in the dataset.
            values: Optional array of unique values for this feature.
                   Type is auto-detected from numpy dtype.
        """
        self._name = name
        self._values = values
        self._missing_value_objects = []
        self._type = self._set_type(values)

    def name(self) -> str:
        """Return the feature name.

        Returns:
            Feature name string.
        """
        return self._name

    def values(self) -> List:
        """Return the unique values for this feature.

        Returns:
            Array of unique feature values.
        """
        return self._values

    def type(self) -> str:
        """Return the feature type.

        Returns:
            Either 'Numerical' or 'Nominal', or None if no values set.
        """
        return self._type

    def set_values(self, values: List) -> None:
        """Update the feature values and re-detect type.

        Args:
            values: New array of unique values for this feature.
                   Type will be auto-detected.
        """
        self._values = values
        self._type = self._set_type(values)

    def set_missing_value_objects(self, indexes: List) -> None:
        """Set the list of row indices where this feature has missing values.

        Args:
            indexes: List of integer row indices with missing values.
        """
        self._missing_value_objects = indexes

    def _set_type(self, values: Optional[np.ndarray]) -> str:
        """Automatically detect feature type from numpy dtype.

        Args:
            values: Array of feature values.

        Returns:
            'Numerical' for int/float types, 'Nominal' for others, None if empty.
        """
        if values is None or len(values) == 0:
            return None
        return 'Numerical' if issubclass(values.dtype.type, (np.integer, np.floating)) else 'Nominal'

    @property
    def missing_value_objects(self) -> List:
        """List of row indices where this feature is missing.

        Returns:
            List of integer indices.
        """
        return self._missing_value_objects


