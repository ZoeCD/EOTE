from EOTE.Protocols import *
from EOTE.Utils import Feature


class NumericalFeatureTreeDirector:
    def __init__(self, builder: NumericalFeatureTreeBuilder):
        self.builder = builder

    def build_feature_tree(self, feature: Feature):
        return self.builder.set_tree_regressor().set_anomaly_scorer().build(feature)

