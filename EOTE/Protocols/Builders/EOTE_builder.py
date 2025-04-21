from typing import Protocol, Optional


class EOTEBuilder(Protocol):
    def set_class_verification_method(self):
        ...

    def set_attribute_cleaning_strategy(self):
        ...

    def set_data_imputer(self):
        ...

    def set_categorical_data_encoder(self):
        ...

    def set_categorical_feature_tree_director(self):
        ...

    def set_numerical_feature_tree_director(self):
        ...

    def set_path_shortener(self):
        ...

    def set_output_format(self):
        ...

    def build(self):
        ...



