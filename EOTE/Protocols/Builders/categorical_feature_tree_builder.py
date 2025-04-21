from typing import Protocol


class CategoricalFeatureTreeBuilder(Protocol):
    def set_feature_weight_calculator(self):
        ...

    def set_feature_values_weight_calculator(self):
        ...

    def set_tree_classifier(self):
        ...

    def build(self, feature):
        ...



