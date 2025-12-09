from EOTE.Protocols import *
from typing import Optional


class EOTEDirector:
    """Director for constructing EOTE instances using the Builder pattern.

    Orchestrates the construction process by calling builder methods in the correct
    order to produce a fully configured EOTE instance. This ensures all required
    components are properly initialized.

    The director enforces the following construction sequence:
    1. Class verification (ensures single-class datasets)
    2. Categorical feature tree director
    3. Attribute cleaning (removes insufficient categorical values)
    4. Data imputation (handles missing values)
    5. Categorical data encoding (OneHot encoding)
    6. Numerical feature tree director
    7. Decision path shortener
    8. Output formatter (terminal or file)

    Args:
        builder: An EOTEBuilder implementation (e.g., EoteWithMissForestInTerminalBuilder)
                that defines how each component should be configured.

    Example:
        >>> from EOTE.Directors import EOTEDirector
        >>> from EOTE.Builders import EoteWithMissForestInTerminalBuilder
        >>>
        >>> director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
        >>> eote = director.get_eote()
        >>> eote.train(X_train, y_train)
    """

    def __init__(self, builder: EOTEBuilder):
        """Initialize director with a builder.

        Args:
            builder: EOTEBuilder implementation defining component configuration.
        """
        self.builder = builder

    def get_eote(self):
        """Construct and return a fully configured EOTE instance.

        Calls builder methods in the required sequence to ensure proper
        component initialization.

        Returns:
            Configured EOTE instance ready for training.
        """
        return self.builder.set_class_verification_method()\
            .set_categorical_feature_tree_director()\
            .set_attribute_cleaning_strategy()\
            .set_data_imputer()\
            .set_categorical_data_encoder()\
            .set_numerical_feature_tree_director()\
            .set_path_shortener()\
            .set_output_format()\
            .build()

