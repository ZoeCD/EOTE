from EOTE.Protocols import *
from typing import Optional


class EOTEDirector:
    def __init__(self, builder: EOTEBuilder):
        self.builder = builder

    def get_eote(self):
        return self.builder.set_class_verification_method()\
            .set_categorical_feature_tree_director()\
            .set_attribute_cleaning_strategy()\
            .set_data_imputer()\
            .set_categorical_data_encoder()\
            .set_numerical_feature_tree_director()\
            .set_path_shortener()\
            .set_output_format()\
            .build()

