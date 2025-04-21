from typing import List
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from EOTE.Protocols import DecisionTree
from typing import Optional


class ClassificationTree(DecisionTree):

    def __init__(self, alphas: Optional[List[float]] = None) -> None:
        self.alphas = alphas

    def set_alphas(self, alphas: List[float]) -> None:
        self.alphas = alphas

    def create_and_fit(self, x: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
        classifier = DecisionTreeClassifier(random_state=0)
        if self.alphas:
            for alpha in self.alphas:
                classifier.ccp_alpha = alpha
                classifier.fit(x, y)
                if classifier.get_depth() > 1:
                    return classifier
        else:
            classifier.fit(x, y)
        return classifier

