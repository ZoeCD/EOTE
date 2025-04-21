from EOTE.Protocols import CategoricalFeatureTreeBuilder
from EOTE.Trees import CategoricalFeatureTree, ClassificationTree
from EOTE.Utils import FeatureWeightByAUC, CategoricalFeatureValuesWeightsByRowSum, Feature


class CategoricalFeatureTreeBuilder(CategoricalFeatureTreeBuilder):
    def __init__(self):
        self.tree = CategoricalFeatureTree()

    def set_feature_weight_calculator(self):
        self.tree.feature_weight_calculator = FeatureWeightByAUC()
        return self

    def set_feature_values_weight_calculator(self):
        self.tree.value_weights_calculator = CategoricalFeatureValuesWeightsByRowSum()
        return self

    def set_tree_classifier(self):
        self.tree.tree_creator = ClassificationTree(alphas=[0.025, 0.010, 0.005])
        return self

    def __set_feature(self, feature: Feature):
        self.tree.set_feature(feature)

    def build(self, feature: Feature):
        self.__set_feature(feature)
        return self.tree
