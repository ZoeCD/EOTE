from typing import Protocol


class NumericalFeatureTreeBuilder(Protocol):
    def set_tree_regressor(self):
        ...

    def set_anomaly_scorer(self):
        ...

    def build(self, feature):
        ...

