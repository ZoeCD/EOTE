from typing import List
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from EOTE.Protocols import DecisionTree
from typing import Optional


class RegressionTree(DecisionTree):

    def __init__(self, alphas: Optional[List[float]] = None) -> None:
        self.alphas = alphas

    def set_alphas(self, alphas: List[float]) -> None:
        self.alphas = alphas

    def create_and_fit(self, x: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
        regressor = DecisionTreeRegressor(random_state=0)
        if self.alphas:
            for alpha in self.alphas:
                regressor.ccp_alpha = alpha
                regressor.fit(x, y)
                if regressor.get_depth() > 1:
                    return regressor
        else:
            regressor.fit(x, y)
        return regressor

