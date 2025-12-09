import pandas as pd
from typing import List, Union
from EOTE.Utils import Feature
from pandas.api.types import is_string_dtype
import numpy as np
from sklearn.tree import _tree
from re import match
from colorama import Fore, Style
from EOTE.Protocols import FeatureTree
import warnings
warnings.filterwarnings("ignore")


class EOTE:
    """Explainable Outlier Tree-based AutoEncoder for anomaly detection.

    EOTE is an anomaly detection system for tabular data that provides interpretable,
    rule-based explanations for its predictions. It uses a per-feature autoencoding
    approach where each feature is predicted from all other features using decision trees.

    The algorithm:
    1. Trains one decision tree per feature to predict that feature from all others
    2. Calculates normal and anomaly scores based on prediction accuracy
    3. Provides interpretable decision rules explaining why instances are classified
       as normal or anomalous

    Attributes:
        class_verification_method: Verifies dataset has single class (semi-supervised)
        attribute_remover: Removes attributes with insufficient categorical values
        imputer: Handles missing data imputation (e.g., MissForest)
        encoder: Encodes categorical features (e.g., OneHotEncoder)
        path_shortener: Simplifies decision tree paths for readability
        cat_feature_tree_director: Director for building categorical feature trees
        cat_feature_tree_builder: Builder for categorical feature trees
        num_feature_tree_director: Director for building numerical feature trees
        num_feature_tree_builder: Builder for numerical feature trees
        output_formatting: Formatter for classification results output

    Note:
        EOTE should be constructed using the Builder pattern via EOTEDirector,
        not directly instantiated. See example_terminal_output.py for usage.

    Example:
        >>> from EOTE.Directors import EOTEDirector
        >>> from EOTE.Builders import EoteWithMissForestInTerminalBuilder
        >>>
        >>> director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
        >>> eote = director.get_eote()
        >>> eote.train(X_train, y_train)
        >>> predictions = eote.classify(X_test)
        >>> eote.classify_and_interpret(X_test.loc[0])

    References:
        D. L. Aguilar et al., "Towards an interpretable autoencoder: A decision
        tree-based autoencoder and its application in anomaly detection,"
        IEEE Transactions on Dependable and Secure Computing, 2022.
    """

    def __init__(self):
        """Initialize EOTE with empty configuration.

        All components (imputer, encoder, directors, etc.) must be set via
        Builder pattern before use.
        """
        self.class_verification_method = None
        self.attribute_remover = None
        self.imputer = None
        self.encoder = None
        self.path_shortener = None
        self.cat_feature_tree_director = None
        self.cat_feature_tree_builder = None
        self.num_feature_tree_director = None
        self.num_feature_tree_builder = None
        self.output_formatting = None


        self.x = None
        self.__original_feature_names = None
        self.features = None
        self.__numerical_columns = list()
        self.__categorical_columns = list()
        self.__x_encoded = None
        self.__feature_trees = list()
        self.__is_trained = False


    def train(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Train the EOTE model on labeled data.

        Trains one decision tree per feature to predict that feature from all others.
        The model learns patterns of normal behavior for anomaly detection.

        Args:
            x: Feature DataFrame with shape (n_samples, n_features).
               Supports mixed numerical and categorical features.
               Missing values will be imputed using the configured imputer.
            y: Target Series with shape (n_samples,). Must contain a single class
               for semi-supervised anomaly detection (all training samples should
               be from the "normal" class).

        Raises:
            ValueError: If the dataset contains multiple classes or if the class
                       verification method detects invalid data.
            Exception: If no feature trees can be trained due to insufficient
                      variability in the features.

        Note:
            After imputation, decision rules should be interpreted with care as
            they may reference imputed values rather than original data.

        Example:
            >>> eote = director.get_eote()
            >>> X_train = pd.DataFrame({'age': [25, 30, 35], 'income': [50k, 60k, 70k]})
            >>> y_train = pd.Series(['normal', 'normal', 'normal'])
            >>> eote.train(X_train, y_train)
        """
        self.__preprocess_dataset(x, y)
        self.__train_trees()
        self.__check_variability()
        self.__is_trained = True

    def __preprocess_dataset(self, x: pd.DataFrame, y: pd.Series) -> None:
        if self.class_verification_method.verify_if_valid(y):
            self.__original_feature_names = list(x.columns)
            self.x = self.attribute_remover.remove_invalid_attributes(x)
            self.features = self.__create_features()
            self.x = self.imputer.fit_transform(self.x)
            self.__divide_columns_by_type()
            self.encoder.fit(self.x[self.__categorical_columns])
            encoded_values = pd.DataFrame(self.encoder.transform(self.x[self.__categorical_columns]),
                                                       columns=self.encoder.get_feature_names_out())
            self.__x_encoded = pd.concat([self.x[self.__numerical_columns], encoded_values], axis=1)


    def __create_features(self) -> List[Feature]:
        features = list()
        for attr_name in self.x.columns:
            values = self.x[attr_name].unique()
            if pd.isnull(values).any():
                values = values[~pd.isnull(values)]

            feature = Feature(attr_name, values)
            feature.set_missing_value_objects(self.x[self.x[attr_name].isnull()].index.to_list())
            features.append(feature)
        return features

    def __divide_columns_by_type(self) -> None:
        self.__categorical_columns = [column for column in self.x.columns if is_string_dtype(self.x[column])]
        self.__numerical_columns = [column for column in self.x.columns if not is_string_dtype(self.x[column])]

    def __train_trees(self) -> None:
        for feature in self.features:
            current_x = self.__get_feature_x(feature)
            current_y = self.__get_feature_y(feature)
            if feature.type() == 'Nominal':
                feature_tree = self.cat_feature_tree_director(self.cat_feature_tree_builder()).build_feature_tree(feature)
                feature_tree.fit(current_x, current_y)
            else:
                feature_tree = self.num_feature_tree_director(self.num_feature_tree_builder()).build_feature_tree(feature)
                feature_tree.fit(current_x, current_y)
            self.__feature_trees.append(feature_tree)

    def __get_feature_x(self, feature: Feature) -> None:
        feature_columns = [v for v in self.__x_encoded.columns if v.startswith(feature.name())]
        new_data = self.__x_encoded.drop(columns=feature_columns)
        new_data = new_data.drop(index=feature.missing_value_objects)
        return new_data

    def __get_feature_y(self, feature: Feature) -> pd.DataFrame:
        new_data = self.x[feature.name()].drop(index=feature.missing_value_objects)
        return new_data

    def __check_variability(self) -> None:
        if len(self.__feature_trees) == 0:
            raise Exception("Unable to train: Not enough variability of the features")

    def classify(self, instances: pd.DataFrame) -> List[List[float]]:
        """Classify instances as normal or anomalous.

        Calculates anomaly scores for each instance by comparing predicted vs actual
        feature values across all trained feature trees. Returns numerical scores
        where positive values indicate anomalies and values near zero indicate normal.

        Args:
            instances: DataFrame with shape (n_samples, n_features) containing
                      instances to classify. Must have the same features as training data.
                      Unknown categorical values will be treated as missing and imputed.

        Returns:
            List of lists, where each inner list contains a single float anomaly score.
            - Score > 0: Instance is classified as anomalous
            - Score ≈ 0: Instance is classified as normal
            - Higher positive scores indicate stronger anomaly signals

        Raises:
            ValueError: If the model has not been trained yet.

        Example:
            >>> scores = eote.classify(X_test)
            >>> print(scores)
            [[0.0], [-0.15], [0.82], ...]  # First is normal, third is anomaly
        """
        self.__check_if_trained()
        instances_known_values = self.__replace_unknown_values(instances)
        instances_encoded = self.__encode_instances(self.imputer.transform(instances_known_values))
        results = list()
        for i in range(len(instances.values)):
            results.append(self.__classify_instance(instances.iloc[i], instances_encoded.iloc[i]))
        return results

    def __check_if_trained(self) -> None:
        if not self.__is_trained:
            raise ValueError("Unable to classify: Untrained DTAE!")

    def __replace_unknown_values(self, instances: pd.DataFrame) -> pd.DataFrame:
        instances = instances[self.__original_feature_names]
        for feature in self.features:
            if feature.type() == 'Nominal':
                to_replace = [
                    value for value in instances[feature.name()].unique()
                    if value not in feature.values()
                ]
                instances[feature.name()].replace(to_replace, value=np.nan, inplace=True)
        return instances

    def __encode_instances(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded_values = pd.DataFrame(self.encoder.transform(data[self.__categorical_columns]),
                                      columns=self.encoder.get_feature_names_out())
        encoded_data = pd.concat([data[self.__numerical_columns], encoded_values], axis=1)
        return encoded_data

    def __classify_instance(self, instance: pd.Series, encoded_instance: pd.Series) -> list:
        score_normal, score_anomaly = self.__classify_per_feature(instance, encoded_instance)
        total_score = score_normal + score_anomaly
        if total_score > 0:
            return [score_anomaly - score_normal]
        else:
            return [0.0]

    def __classify_per_feature(self, instance: pd.Series, encoded_instance: pd.Series) -> (float, float):
        score_anomaly, score_normal = 0.0, 0.0
        for feature_tree in self.__feature_trees:
            if not pd.isnull(instance[feature_tree.feature.name()]):
                current_instance = self.__get_instance_x(feature_tree.feature, encoded_instance)
                score_normal += feature_tree.calculate_score_normal(current_instance, instance[feature_tree.feature.name()])
                score_anomaly += feature_tree.calculate_score_anomaly(current_instance, instance[feature_tree.feature.name()])
        return score_normal, score_anomaly

    def __get_instance_x(self, feature: Feature, instance: pd.Series) -> pd.Series:
        feature_columns = list(filter(lambda v: match(f'{feature.name()}.*', v), instance.index))
        return instance.drop(index=feature_columns)

    def classify_and_interpret(self, instance: pd.Series) -> None:
        """Classify a single instance and output human-readable explanation rules.

        Classifies an instance and generates interpretable decision rules explaining
        why the instance is classified as normal or anomalous. The output is formatted
        according to the configured output formatter (terminal or file).

        Args:
            instance: A pandas Series representing a single instance to classify.
                     Must contain all features from the training data.
                     Index should be feature names matching training data.

        Raises:
            ValueError: If the model has not been trained yet.

        Note:
            This method produces output via the configured output formatter (either
            to terminal with colors or to a text file). The output includes:
            - Overall classification (Normal/Anomaly)
            - Final anomaly score
            - Decision rules supporting the anomaly classification
            - Decision rules supporting the normal classification

        Example:
            >>> eote.classify_and_interpret(X_test.loc[0])
            # Output (to terminal or file):
            # Instance: [age=45, income=120000, ...]
            # Classification: Anomaly (score: 0.82)
            # Anomaly Rules:
            #   - If (age > 40) AND (income > 100000) then (status = unemployed)
            # Normal Rules:
            #   - If (education ≤ 12) then (credit_score = 650)
        """
        self.__check_if_trained()
        instance_as_dataframe = instance.to_frame().T.reset_index()
        instance_known_values = self.__replace_unknown_values(instance_as_dataframe)
        instance_encode = self.__encode_instances(self.imputer.transform(instance_known_values))
        instance_encode = instance_encode.iloc[0]
        anomaly_rules, normal_rules = list(), list()

        score_anomaly, score_normal = 0, 0
        for feature_tree in self.__feature_trees:
            real_value = instance[feature_tree.feature.name()]
            if not pd.isnull(real_value):
                current_instance = self.__get_instance_x(feature_tree.feature, instance_encode)
                normal = feature_tree.calculate_score_normal(current_instance, real_value)
                anomaly = feature_tree.calculate_score_anomaly(current_instance, real_value)
                current_instance = current_instance.to_frame().T
                score_normal += normal
                score_anomaly += anomaly
                if feature_tree.feature.type() == 'Nominal':
                    classifier_result = feature_tree.classifier.predict(current_instance)[0]
                else:
                    classifier_result = feature_tree.regressor.predict(current_instance)[0]
                decision_path = self.__get_decision_path(feature_tree, current_instance, classifier_result)
                decision_path = self.path_shortener.shorten_path(self.__original_feature_names,
                                                                 current_instance.columns, decision_path)

                if normal - anomaly > 0:
                    normal_rules.append(decision_path)
                else:
                    anomaly_rules.append(decision_path)

        total_score = score_normal + score_anomaly
        final_score = 0.0
        if total_score > 0:
            final_score = score_normal - score_anomaly

        self.output_formatting.produce_output(instance, final_score, anomaly_rules, normal_rules, self.__original_feature_names)

    def __get_decision_path(self, feature_tree: FeatureTree, instance: pd.DataFrame, real_value: Union[str, float]) -> str:
        # By:  https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

        tree = feature_tree.classifier if feature_tree.feature.type() == 'Nominal' else feature_tree.regressor
        feature_names = instance.columns
        sample_id = 0
        tree_ = tree.tree_

        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        node_indicator = tree.decision_path(instance)
        leaf_id = tree.apply(instance)

        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

        result = f"({feature_tree.feature.name()} = {real_value})"
        decision_path = ""

        if feature_tree.feature.type() == 'Numerical':
            samples_leaf = feature_tree.leaves_and_instances.get(leaf_id[0])
            if len(samples_leaf) > 1 and len(np.unique(samples_leaf)) > 1:
                result = f"({min(samples_leaf)} ≤ {feature_tree.feature.name()} ≤ {max(samples_leaf)})"

        if tree.get_n_leaves() > 1:
            decision_path += "If "
        else:
            decision_path += f"Classifier has only one node, results are: {result}"

        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                continue

            threshold_sign = ''
            if instance.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "≤"
            else:
                threshold_sign = ">"

            if node_id == node_index[-2]:
                decision_path += f"({feature_name[node_id]} {threshold_sign} {threshold[node_id]}) then {result}"
            else:

                decision_path += f"({feature_name[node_id]} {threshold_sign} {threshold[node_id]}) AND "

        return decision_path
