from typing import Protocol, List
from pandas import Series


class OutputFormatter(Protocol):
    def produce_output(self, instance: Series,
                       final_score: float,
                       anomaly_rules: List[str],
                       normal_rules: List[str],
                       feature_names: List[str]) -> None:
        ...
