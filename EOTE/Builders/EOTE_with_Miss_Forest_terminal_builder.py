from EOTE import EOTE
from EOTE.Protocols import EOTEBuilder
from EOTE.Utils import UniqueClassVerifier,\
    InsufficientCategoricalValuesAttributeRemover, \
    MissForestImputer, \
    TerminalOutputFormatter
from EOTE.Directors import CategoricalFeatureTreeDirector, NumericalFeatureTreeDirector
from EOTE.Trees import PathShortenerMixedData
from .categorical_feature_tree_builder import CategoricalFeatureTreeBuilder
from .numerical_feature_tree_builder import NumericalFeatureTreeBuilder
from sklearn.preprocessing import OneHotEncoder


class EoteWithMissForestInTerminalBuilder(EOTEBuilder):
    """Builder for EOTE with MissForest imputation and terminal output.

    Configures EOTE with:
    - MissForest imputation for handling missing data
    - OneHot encoding for categorical features
    - Classification trees for categorical features (AUC-based weighting)
    - Regression trees for numerical features (domain-based scoring)
    - Terminal output with colored formatting

    This builder uses default hyperparameters suitable for most use cases:
    - CCP alphas for tree pruning: [0.025, 0.010, 0.005]
    - Minimum categorical values per attribute: 2 with 3 instances minimum
    - MissForest with default settings (max_iter=20, tol=0.24)

    Example:
        >>> from EOTE.Directors import EOTEDirector
        >>> director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
        >>> eote = director.get_eote()
        >>> eote.train(X_train, y_train)
        >>> eote.classify_and_interpret(X_test.loc[0])  # Prints to terminal
    """

    def __init__(self):
        """Initialize builder with empty EOTE instance."""
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
