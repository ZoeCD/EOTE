from EOTE import EOTE
from EOTE.Protocols import EOTEBuilder
from EOTE.Utils import UniqueClassVerifier,\
    InsufficientCategoricalValuesAttributeRemover, \
    MissForestImputer, \
    TxtFileOutputFormatter
from EOTE.Ditectors import CategoricalFeatureTreeDirector, NumericalFeatureTreeDirector
from EOTE.Trees import PathShortenerMixedData
from .categorical_feature_tree_builder import CategoricalFeatureTreeBuilder
from .numerical_feature_tree_builder import NumericalFeatureTreeBuilder
from sklearn.preprocessing import OneHotEncoder


class EoteWithMissForestInTxTFileBuilder(EOTEBuilder):
    def __init__(self, output_path: str = "output.txt"):
        self.EOTE = EOTE()
        self.output_path = output_path

    def set_class_verification_method(self):
        self.EOTE.class_verification_method = UniqueClassVerifier()
        return self

    def set_attribute_cleaning_strategy(self):
        self.EOTE.attribute_remover = InsufficientCategoricalValuesAttributeRemover(2,3)
        return self

    def set_data_imputer(self):
        self.EOTE.imputer = MissForestImputer()
        return self

    def set_categorical_data_encoder(self):
        self.EOTE.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return self

    def set_categorical_feature_tree_director(self):
        self.EOTE.cat_feature_tree_director = CategoricalFeatureTreeDirector
        self.EOTE.cat_feature_tree_builder = CategoricalFeatureTreeBuilder
        return self

    def set_numerical_feature_tree_director(self):
        self.EOTE.num_feature_tree_director = NumericalFeatureTreeDirector
        self.EOTE.num_feature_tree_builder = NumericalFeatureTreeBuilder
        return self

    def set_path_shortener(self):
        self.EOTE.path_shortener = PathShortenerMixedData()
        return self

    def set_output_format(self):
        self.EOTE.output_formatting = TxtFileOutputFormatter(self.output_path)
        return self

    def build(self):
        return self.EOTE
