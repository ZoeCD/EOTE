from EOTE.Protocols import NumericalFeatureTreeBuilder
from EOTE.Trees import NumericalFeatureTree, RegressionTree
from EOTE.Utils import Feature, AnomalyDomainScorer


class NumericalFeatureTreeBuilder(NumericalFeatureTreeBuilder):
    def __init__(self):
        self.tree = NumericalFeatureTree()

    def set_tree_regressor(self):
        self.tree.regressor = RegressionTree(alphas=[0.025, 0.010, 0.005])
        return self

    def set_anomaly_scorer(self):
        self.tree.anomaly_scorer = AnomalyDomainScorer()
        return self

    def __set_feature(self, feature: Feature):
        self.tree.set_feature(feature)

    def build(self, feature: Feature):
        self.__set_feature(feature)
        return self.tree
