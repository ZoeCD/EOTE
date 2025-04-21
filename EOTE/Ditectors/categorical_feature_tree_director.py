from EOTE.Protocols import *
from EOTE.Utils import Feature


class CategoricalFeatureTreeDirector:
    def __init__(self, builder: CategoricalFeatureTreeBuilder):
        self.builder = builder

    def build_feature_tree(self, feature: Feature):
        return self.builder.set_tree_classifier()\
            .set_feature_weight_calculator()\
            .set_feature_values_weight_calculator()\
            .build(feature)

