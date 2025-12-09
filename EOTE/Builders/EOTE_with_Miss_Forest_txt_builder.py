from EOTE import EOTE
from EOTE.Protocols import EOTEBuilder
from EOTE.Utils import UniqueClassVerifier,\
    InsufficientCategoricalValuesAttributeRemover, \
    MissForestImputer, \
    TxtFileOutputFormatter
from EOTE.Directors import CategoricalFeatureTreeDirector, NumericalFeatureTreeDirector
from EOTE.Trees import PathShortenerMixedData
from .categorical_feature_tree_builder import CategoricalFeatureTreeBuilder
from .numerical_feature_tree_builder import NumericalFeatureTreeBuilder
from sklearn.preprocessing import OneHotEncoder


class EoteWithMissForestInTxTFileBuilder(EOTEBuilder):
    """Builder for EOTE with MissForest imputation and text file output.

    Configures EOTE with:
    - MissForest imputation for handling missing data
    - OneHot encoding for categorical features
    - Classification trees for categorical features (AUC-based weighting)
    - Regression trees for numerical features (domain-based scoring)
    - Text file output for classification results

    This builder uses default hyperparameters suitable for most use cases:
    - CCP alphas for tree pruning: [0.025, 0.010, 0.005]
    - Minimum categorical values per attribute: 2 with 3 instances minimum
    - MissForest with default settings (max_iter=20, tol=0.24)

    Args:
        output_path: Path to the output text file where classification results
                    will be written. Defaults to "output.txt".

    Example:
        >>> from EOTE.Directors import EOTEDirector
        >>> builder = EoteWithMissForestInTxTFileBuilder("results.txt")
        >>> director = EOTEDirector(builder)
        >>> eote = director.get_eote()
        >>> eote.train(X_train, y_train)
        >>> eote.classify_and_interpret(X_test.loc[0])  # Writes to results.txt
    """

    def __init__(self, output_path: str = "output.txt"):
        """Initialize builder with empty EOTE instance and output path.

        Args:
            output_path: Path where classification results will be written.
        """
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
