from EOTE.Protocols import OutputFormatter
from typing import List
from pandas import Series
from colorama import Fore, Style


class TerminalOutputFormatter(OutputFormatter):

    def produce_output(self, instance: Series,
                       final_score: float,
                       anomaly_rules: List[str],
                       normal_rules: List[str],
                       feature_names: List[str]) -> None:
        print(f"Given the following object: ")
        string_object = ", ".join(
            [f"{feature} = {value}" for feature, value in zip(feature_names, instance)])
        print(string_object)
        print()

        if final_score > 0:
            print(f"The object is classified as: " + Style.BRIGHT + "Normal")
            result = "Normal"
        else:
            print(f"The object is classified as: " + Style.BRIGHT + "Anomaly")
            result = "Anomaly"

        print(Style.RESET_ALL)
        print()

        print("The following rules votes for the object to be normal: ")
        for rule in normal_rules:
            print(rule)

        print()

        print("The following rules votes for the object to be an anomaly: ")
        for rule in anomaly_rules:
            print(rule)
            
        return result
