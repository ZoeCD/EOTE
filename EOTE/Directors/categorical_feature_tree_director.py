from EOTE.Protocols import *
from EOTE.Utils import Feature


class CategoricalFeatureTreeDirector:
    """Director for constructing categorical feature trees.

    Orchestrates the construction of a CategoricalFeatureTree by calling builder
    methods in the correct sequence. Each categorical feature tree is configured to:
    1. Use a classification tree for prediction
    2. Calculate feature importance using AUC
    3. Calculate categorical value weights

    Args:
        builder: A CategoricalFeatureTreeBuilder implementation.

    Example:
        >>> director = CategoricalFeatureTreeDirector(CategoricalFeatureTreeBuilder())
        >>> feature = Feature(name="color", values=["red", "blue", "green"])
        >>> tree = director.build_feature_tree(feature)
        >>> tree.fit(X, y)
    """

    def __init__(self, builder: CategoricalFeatureTreeBuilder):
        """Initialize director with a categorical feature tree builder.

        Args:
            builder: CategoricalFeatureTreeBuilder for constructing the tree.
        """
        self.builder = builder

    def build_feature_tree(self, feature: Feature):
        """Build and return a configured categorical feature tree.

        Args:
            feature: Feature metadata for the categorical feature to predict.

        Returns:
            Configured CategoricalFeatureTree ready for training.
        """
        return self.builder.set_tree_classifier()\
            .set_feature_weight_calculator()\
            .set_feature_values_weight_calculator()\
            .build(feature)

