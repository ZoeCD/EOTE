from EOTE.Protocols import NumericalFeatureTreeBuilder
from EOTE.Trees import NumericalFeatureTree, RegressionTree
from EOTE.Utils import Feature, AnomalyDomainScorer


class NumericalFeatureTreeBuilder(NumericalFeatureTreeBuilder):
    """Builder for numerical feature trees.

    Constructs a NumericalFeatureTree configured with:
    - Regression tree with cost-complexity pruning (CCP alphas: [0.025, 0.010, 0.005])
    - Domain-based anomaly scorer (normalized by feature value range)

    The numerical feature tree predicts a numerical feature from all other features,
    then calculates normal/anomaly scores based on prediction error relative to
    the feature's domain range.

    Example:
        >>> builder = NumericalFeatureTreeBuilder()
        >>> feature_tree = builder.set_tree_regressor() \\
        ...                       .set_anomaly_scorer() \\
        ...                       .build(feature)
    """

    def __init__(self):
        """Initialize builder with empty NumericalFeatureTree."""
        self.tree = NumericalFeatureTree()

    def set_tree_regressor(self):
        """Set regression tree with CCP pruning.

        Uses CCP alphas [0.025, 0.010, 0.005] for cost-complexity pruning.

        Returns:
            Self for method chaining.
        """
        self.tree.regressor = RegressionTree(alphas=[0.025, 0.010, 0.005])
        return self

    def set_anomaly_scorer(self):
        """Set domain-based anomaly scorer.

        Scores are normalized by the feature's value range (max - min).

        Returns:
            Self for method chaining.
        """
        self.tree.anomaly_scorer = AnomalyDomainScorer()
        return self

    def __set_feature(self, feature: Feature):
        """Set the feature to be predicted by this tree.

        Args:
            feature: Feature metadata including name, values, and type.
        """
        self.tree.set_feature(feature)

    def build(self, feature: Feature):
        """Build and return the configured numerical feature tree.

        Args:
            feature: Feature to be predicted by this tree.

        Returns:
            Configured NumericalFeatureTree ready for training.
        """
        self.__set_feature(feature)
        return self.tree
