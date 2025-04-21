from EOTE import EOTE
from EOTE.Protocols import EOTEBuilder
from EOTE.Utils import UniqueClassVerifier,\
    InsufficientCategoricalValuesAttributeRemover, \
    MissForestImputer, \
    TerminalOutputFormatter
from EOTE.Ditectors import CategoricalFeatureTreeDirector, NumericalFeatureTreeDirector
from EOTE.Trees import PathShortenerMixedData
from .categorical_feature_tree_builder import CategoricalFeatureTreeBuilder
from .numerical_feature_tree_builder import NumericalFeatureTreeBuilder
from sklearn.preprocessing import OneHotEncoder


class EoteWithMissForestInTerminalBuilder(EOTEBuilder):
    def __init__(self):
        self.Dtae = EOTE()

    def set_class_verification_method(self):
        self.Dtae.class_verification_method = UniqueClassVerifier()
        return self

    def set_attribute_cleaning_strategy(self):
        self.Dtae.attribute_remover = InsufficientCategoricalValuesAttributeRemover(2,3)
        return self

    def set_data_imputer(self):
        self.Dtae.imputer = MissForestImputer()
        return self

    def set_categorical_data_encoder(self):
        self.Dtae.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return self

    def set_categorical_feature_tree_director(self):
        self.Dtae.cat_feature_tree_director = CategoricalFeatureTreeDirector
        self.Dtae.cat_feature_tree_builder = CategoricalFeatureTreeBuilder
        return self

    def set_numerical_feature_tree_director(self):
        self.Dtae.num_feature_tree_director = NumericalFeatureTreeDirector
        self.Dtae.num_feature_tree_builder = NumericalFeatureTreeBuilder
        return self

    def set_path_shortener(self):
        self.Dtae.path_shortener = PathShortenerMixedData()
        return self

    def set_output_format(self):
        self.Dtae.output_formatting = TerminalOutputFormatter()
        return self

    def build(self):
        return self.Dtae
