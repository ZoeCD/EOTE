import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Dict
from EOTE.Protocols import ImputerInterface


class MissForestImputer(ImputerInterface):
    def __init__(self):
        self.__categorical_features = None
        self.__categorical_encoders = None
        self.__categorical_features_indices = None
        self.iterative_categorical_imputer = None
        self.__numerical_features = None
        self.__numerical_features_indices = None
        self.iterative_numerical_imputer = None
        self.__fitted = False
        self.__non_missing_values_per_col = None
    
    def get_categorical_features(self) -> List[str]:
        return self.__categorical_features

    def get_numerical_features(self) -> List[str]:
        return self.__numerical_features

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def fit(self, data: pd.DataFrame) -> None:
        self.__check_if_dataset_empty(data)
        self.__categorical_features = self.__get_categorical_features(data)
        self.__numerical_features = self.__get_numerical_features(data)
        if self.__categorical_features:
            self.__fit_categorical_attributes(data)
        if self.__numerical_features:
            self.__fit_numerical_attributes(data)
        self.__fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.__check_if_fitted()
        dtypes = data.dtypes.to_dict()
        if self.__categorical_features:
            data = self.__transform_categorical_attributes(data)
        if self.__numerical_features:
            data = self.__transform_numerical_attributes(data)
        return data.astype(dtype=dtypes)

    def __fit_categorical_attributes(self, data: pd.DataFrame) -> None:
        self.__categorical_features_indices = [data.columns.get_loc(col) for col in self.__categorical_features]
        self.__categorical_encoders = self.__get_fit_encoders(data)
        self.__non_missing_values_per_col = self.__get_non_missing_values(data)
        encoded_data = self.__transform_data_with_encoder(data.copy(), self.__non_missing_values_per_col)
        self.iterative_categorical_imputer = self.__create_categorical_iterative_imputer()
        self.iterative_categorical_imputer.fit(encoded_data[self.__categorical_features],
                                               self.__categorical_features_indices)

    def __fit_numerical_attributes(self, data: pd.DataFrame) -> None:
        self.iterative_numerical_imputer = self.__create_numerical_iterative_imputer()
        self.__numerical_features_indices = [data.columns.get_loc(col) for col in self.__categorical_features]
        self.iterative_numerical_imputer.fit(data[self.__numerical_features], self.__numerical_features_indices)

    def __get_fit_encoders(self, data: pd.DataFrame) -> Dict[str, LabelEncoder]:
        label_encoders = {}
        for col in data[self.__categorical_features].columns:
            label_encoder = LabelEncoder()
            label_encoders[col] = label_encoder.fit(data[col].astype(str))
        return label_encoders

    def __transform_data_with_encoder(self, data: pd.DataFrame, non_missing_values_per_col) -> pd.DataFrame:
        for col in self.__categorical_features:
            if data[col].dtype == 'object':
                label_encoder = self.__categorical_encoders.get(col)
                indices = non_missing_values_per_col.get(col)
                data.loc[indices, col] = label_encoder.transform(data.loc[indices, col].astype(str))
        return data

    def __get_categorical_features(self, data: pd.DataFrame) -> List[str]:
        return list(data.select_dtypes(include=['object']).columns)

    def __get_numerical_features(self, data: pd.DataFrame) -> List[str]:
        return list(data.select_dtypes(include=['float64', 'int64']).columns)

    def __create_categorical_iterative_imputer(self):
        imp_cat = IterativeImputer(estimator=RandomForestClassifier(n_jobs=-1, random_state=0),
                                   initial_strategy='most_frequent',
                                   max_iter=20, random_state=0, skip_complete=True, tol=0.24)
        return imp_cat

    def __create_numerical_iterative_imputer(self) -> IterativeImputer:
        imp_num = IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1, random_state=0),
                                   initial_strategy='mean',
                                   max_iter=20, random_state=0, skip_complete=True, tol=0.24)
        return imp_num

    def __transform_categorical_attributes(self, data: pd.DataFrame) -> pd.DataFrame:
        non_missing_values_per_col = self.__get_non_missing_values(data)
        data = self.__transform_data_with_encoder(data, non_missing_values_per_col)
        data[self.__categorical_features] = self.iterative_categorical_imputer.transform(data[self.__categorical_features])
        data[self.__categorical_features] = self.__decode_information(data[self.__categorical_features])
        return data

    def __transform_numerical_attributes(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.__numerical_features] = self.iterative_numerical_imputer.transform(data[self.__numerical_features])
        return data

    def __decode_information(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data[self.__categorical_features].columns:
            label_encoder = self.__categorical_encoders.get(col)
            data[col] = label_encoder.inverse_transform(data[col].astype(int))
        return data

    def __check_if_dataset_empty(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Empty dataset")

    def __check_if_fitted(self):
        if not self.__fitted:
            raise ValueError("Miss Forest Imputer not fitted")

    def __get_non_missing_values(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        missing_values_per_col = {}
        for col in self.__categorical_features:
            missing_values = np.where(~data[col].isnull())[0]
            missing_values_per_col[col] = missing_values
        return missing_values_per_col

