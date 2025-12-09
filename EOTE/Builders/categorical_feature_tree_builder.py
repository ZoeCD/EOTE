from EOTE.Protocols import CategoricalFeatureTreeBuilder
from EOTE.Trees import CategoricalFeatureTree, ClassificationTree
from EOTE.Utils import FeatureWeightByAUC, CategoricalFeatureValuesWeightsByRowSum, Feature


class CategoricalFeatureTreeBuilder(CategoricalFeatureTreeBuilder):
    """Builder for categorical feature trees.

    Constructs a CategoricalFeatureTree configured with:
    - AUC-based feature weight calculation (multiclass AUC from confusion matrix)
    - Row-sum based categorical value weights
    - Classification tree with cost-complexity pruning (CCP alphas: [0.025, 0.010, 0.005])

    The categorical feature tree predicts a categorical feature from all other features,
    then calculates normal/anomaly scores based on prediction accuracy and value weights.

    Example:
        >>> builder = CategoricalFeatureTreeBuilder()
        >>> feature_tree = builder.set_feature_weight_calculator() \\
        ...                       .set_feature_values_weight_calculator() \\
        ...                       .set_tree_classifier() \\
        ...                       .build(feature)
    """

    def __init__(self):
        """Initialize builder with empty CategoricalFeatureTree."""
        self.tree = CategoricalFeatureTree()

    def set_feature_weight_calculator(self):
        """Set AUC-based feature weight calculator.

        Returns:
            Self for method chaining.
        """
        self.tree.feature_weight_calculator = FeatureWeightByAUC()
        return self

    def set_feature_values_weight_calculator(self):
        """Set row-sum based categorical value weights calculator.

        Returns:
            Self for method chaining.
        """
        self.tree.value_weights_calculator = CategoricalFeatureValuesWeightsByRowSum()
        return self

    def set_tree_classifier(self):
        """Set classification tree with CCP pruning.

        Uses CCP alphas [0.025, 0.010, 0.005] for cost-complexity pruning.

        Returns:
            Self for method chaining.
        """
        self.tree.tree_creator = ClassificationTree(alphas=[0.025, 0.010, 0.005])
        return self

    def __set_feature(self, feature: Feature):
        """Set the feature to be predicted by this tree.

        Args:
            feature: Feature metadata including name, values, and type.
        """
        self.tree.set_feature(feature)

    def build(self, feature: Feature):
        """Build and return the configured categorical feature tree.

        Args:
            feature: Feature to be predicted by this tree.

        Returns:
            Configured CategoricalFeatureTree ready for training.
        """
        self.__set_feature(feature)
        return self.tree
