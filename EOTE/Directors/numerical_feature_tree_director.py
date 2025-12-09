from EOTE.Protocols import *
from EOTE.Utils import Feature


class NumericalFeatureTreeDirector:
    """Director for constructing numerical feature trees.

    Orchestrates the construction of a NumericalFeatureTree by calling builder
    methods in the correct sequence. Each numerical feature tree is configured to:
    1. Use a regression tree for prediction
    2. Use domain-based anomaly scoring

    Args:
        builder: A NumericalFeatureTreeBuilder implementation.

    Example:
        >>> director = NumericalFeatureTreeDirector(NumericalFeatureTreeBuilder())
        >>> feature = Feature(name="age", values=[25, 30, 35, 40])
        >>> tree = director.build_feature_tree(feature)
        >>> tree.fit(X, y)
    """

    def __init__(self, builder: NumericalFeatureTreeBuilder):
        """Initialize director with a numerical feature tree builder.

        Args:
            builder: NumericalFeatureTreeBuilder for constructing the tree.
        """
        self.builder = builder

    def build_feature_tree(self, feature: Feature):
        """Build and return a configured numerical feature tree.

        Args:
            feature: Feature metadata for the numerical feature to predict.

        Returns:
            Configured NumericalFeatureTree ready for training.
        """
        return self.builder.set_tree_regressor().set_anomaly_scorer().build(feature)

