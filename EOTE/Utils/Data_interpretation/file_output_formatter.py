from EOTE.Protocols import OutputFormatter
from typing import List
from pandas import Series


class TxtFileOutputFormatter(OutputFormatter):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def produce_output(self, instance: Series,
                       final_score: float,
                       anomaly_rules: List[str],
                       normal_rules: List[str],
                       feature_names: List[str]) -> None:

        text_file = open(self.file_path, "w")
        text_file.write("Given the following object: \n")
        string_object = ", ".join(
            [f"{feature} = {value}" for feature, value in zip(feature_names, instance)])
        text_file.write(string_object)
        text_file.write("\n\n")

        if final_score > 0:
            text_file.write(f"The object is classified as: Normal\n")
        else:
            text_file.write(f"The object is classified as: Anomaly\n")

        text_file.write("\n")

        text_file.write("The following rules votes for the object to be normal:\n")
        for rule in normal_rules:
            text_file.write(rule)
            text_file.write("\n")

        text_file.write("\n\n")

        text_file.write("The following rules votes for the object to be an anomaly:\n")
        for rule in anomaly_rules:
            text_file.write(rule)
            text_file.write("\n")
