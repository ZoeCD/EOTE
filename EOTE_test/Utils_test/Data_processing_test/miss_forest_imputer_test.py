import sys
sys.path.append(".")
from EOTE.Utils.Data_processing import miss_forest_imputer
from EOTE.Utils.Data_processing.miss_forest_imputer import MissForestImputer
import pandas as pd
import numpy as np
import pytest


def test_categorical_imputer_less_missing():
	data = [["yellow", "small", "stretch", np.nan],
			["yellow", "small", np.nan, "adult"],
			[np.nan, "small", "dip", "child"],
			["yellow", "large", np.nan, "child"],
			["yellow", "large", "dip", "adult"],
			["yellow", "large", np.nan, np.nan],
			["purple", "small", "stretch", "child"],
			[np.nan, "small", "dip", "adult"],
			["purple", "small", np.nan, "child"],
			[np.nan, "large", "stretch", np.nan],
			["purple", "large", "dip", "adult"],
			[np.nan, "large", "dip", "child"],
			]
	columns = ["color", "size", "act", "age"]
	dataset = pd.DataFrame(data, columns=columns)

	answer_data = [["yellow", "small", "stretch", "child"],
				["yellow", "small", "dip", "adult"],
				["yellow", "small", "dip", "child"],
				["yellow", "large", "dip", "child"],
				["yellow", "large", "dip", "adult"],
				["yellow", "large", "dip", "child"],
				["purple", "small", "stretch", "child"],
				["yellow", "small", "dip", "adult"],
				["purple", "small", "stretch", "child"],
				["yellow", "large", "stretch", "child"],
				["purple", "large", "dip", "adult"],
				["yellow", "large", "dip", "child"]]
	columns = ["color", "size", "act", "age"]
	expected_dataset = pd.DataFrame(answer_data, columns=columns)

	imp = MissForestImputer()
	actual_dataset = imp.fit_transform(dataset)
	pd.testing.assert_frame_equal(expected_dataset, actual_dataset)

def test_categorical_imputer_most_missing():
	data = [["yellow", np.nan, np.nan, "child"],
			["yellow", np.nan, np.nan, "adult"],
			[np.nan, np.nan, "dip", "child"],
			[np.nan, np.nan, np.nan, "child"],
			[np.nan, np.nan, "dip", "adult"],
			[np.nan, "small", np.nan, "adult"],
			[np.nan, np.nan, np.nan, "child"],
			[np.nan, np.nan, np.nan, "adult"],
			[np.nan, np.nan, np.nan, "child"],
			[np.nan, np.nan, np.nan, "child"],
			[np.nan, np.nan, "dip", "adult"],
			[np.nan, np.nan, "dip", "child"],
			]
	columns = ["color", "size", "act", "age"]
	dataset = pd.DataFrame(data, columns=columns)

	data = [["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "adult"],
			["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "adult"],
			["yellow", "small", "dip", "adult"],
			["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "adult"],
			["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "child"],
			["yellow", "small", "dip", "adult"],
			["yellow", "small", "dip", "child"],
			]
	columns = ["color", "size", "act", "age"]
	expected_dataset = pd.DataFrame(data, columns=columns)

	imp = MissForestImputer()
	actual_dataset = imp.fit_transform(dataset)
	pd.testing.assert_frame_equal(expected_dataset, actual_dataset)

def test_categorical_imputer_empty_dataset():
	data = []
	dataset = pd.DataFrame(data)
	imp = MissForestImputer()

	with pytest.raises(ValueError):
			actual_dataset = imp.fit_transform(dataset)

def test_categorical_imputer_fit_then_transform():
	data = [["yellow", "small", "stretch", np.nan],
			["yellow", "small", np.nan, "adult"],
			[np.nan, "small", "dip", "child"],
			["yellow", "large", np.nan, "child"],
			["yellow", "large", "dip", "adult"],
			["yellow", "large", np.nan, np.nan],
			["purple", "small", "stretch", "child"],
			[np.nan, "small", "dip", "adult"],
			["purple", "small", np.nan, "child"],
			[np.nan, "large", "stretch", np.nan],
			["purple", "large", "dip", "adult"],
			[np.nan, "large", "dip", "child"],
			]
	columns = ["color", "size", "act", "age"]
	dataset = pd.DataFrame(data, columns=columns)

	answer_data = [["yellow", "small", "stretch", "child"],
					["yellow", "small", "dip", "adult"],
					["yellow", "small", "dip", "child"],
					["yellow", "large", "dip", "child"],
					["yellow", "large", "dip", "adult"],
					["yellow", "large", "dip", "child"],
					["purple", "small", "stretch", "child"],
					["yellow", "small", "dip", "adult"],
					["purple", "small", "stretch", "child"],
					["yellow", "large", "stretch", "child"],
					["purple", "large", "dip", "adult"],
					["yellow", "large", "dip", "child"]]
	expected_dataset = pd.DataFrame(answer_data, columns=columns)

	imp = MissForestImputer()
	imp.fit(dataset)
	actual_dataset = imp.transform(dataset)
	pd.testing.assert_frame_equal(expected_dataset, actual_dataset)

def test_categorical_imputer_transform_without_fit():
	data = [["yellow", "small", "stretch", "birthday", np.nan],
			["yellow", "small", np.nan, "graduation", "adult"],
			[np.nan, "small", "dip", "birthday", "child"],
			["yellow", "large", np.nan, "birthday", "child"],
			["yellow", "large", "dip", "birthday", "adult"],
			["yellow", "large", np.nan, "graduation", np.nan],
			["purple", "small", "stretch", "birthday", "child"],
			[np.nan, "small", "dip", "birthday", "adult"],
			["purple", "small", np.nan, "birthday", "child"],
			[np.nan, "large", "stretch", "birthday", np.nan],
			["purple", "large", "dip", "graduation", "adult"],
			[np.nan, "large", "dip", "birthday", "child"],
			]
	columns = ["color", "size", "act", "theme", "age"]
	dataset = pd.DataFrame(data, columns=columns)

	imp = MissForestImputer()

	with pytest.raises(ValueError):
			actual_dataset = imp.transform(dataset)

def test_numerical_imputer_less_missing():
	data = [[0.2323, 0.5738, 0.3243, np.nan],
			[0.2374, 0.5738, np.nan, 0.9823],
			[np.nan, 0.5738, 0.3724, 0.9820],
			[0.2375, 0.5734, np.nan, 0.9825],
			[0.2405, 0.5738, 0.3592, 0.9824],
			[0.2322, 0.5739, np.nan, np.nan],
			[0.0323, 0.5736, 0.3350, 0.9828],
			]
	dataset = pd.DataFrame(data, columns=["col1", "col2", "col3", "col4"])

	answer_data = [[0.2323, 0.5738, 0.3243,  0.9823],
					[0.2374, 0.5738, 0.3581, 0.9823],
					[0.2391, 0.5738, 0.3724, 0.9820],
					[0.2375, 0.5734, 0.3452, 0.9825],
					[0.2405, 0.5738, 0.3592, 0.9824],
					[0.2322, 0.5739, 0.3492,  0.9823],
					[0.0323, 0.5736, 0.3350, 0.9828],
					]

	expected_dataset = pd.DataFrame(answer_data, columns=["col1", "col2", "col3", "col4"])

	imp = MissForestImputer()
	actual_dataset = imp.fit_transform(dataset)
	pd.testing.assert_frame_equal(expected_dataset, actual_dataset, rtol=0.24, atol=0.24)

def test_numerical_imputer_fit_then_transform():
	data = [[0.2323, 0.5738, 0.3243, np.nan],
			[0.2374, 0.5738, np.nan, 0.9823],
			[np.nan, 0.5738, 0.3724, 0.9820],
			[0.2375, 0.5734, np.nan, 0.9825],
			[0.2405, 0.5738, 0.3592, 0.9824],
			[0.2322, 0.5739, np.nan, np.nan],
			[0.0323, 0.5736, 0.3350, 0.9828],
			]
	columns = ["col1", "col2", "col3", "col4"]
	dataset = pd.DataFrame(data, columns=["col1", "col2", "col3", "col4"])

	answer_data = [[0.2323, 0.5738, 0.3243, 0.9823],
					[0.2374, 0.5738, 0.3581, 0.9823],
					[0.2391, 0.5738, 0.3724, 0.9820],
					[0.2375, 0.5734, 0.3452, 0.9825],
					[0.2405, 0.5738, 0.3592, 0.9824],
					[0.2322, 0.5739, 0.3492, 0.9823],
					[0.0323, 0.5736, 0.3350, 0.9828],
					]

	expected_dataset = pd.DataFrame(answer_data, columns=["col1", "col2", "col3", "col4"])

	imp = MissForestImputer()
	imp.fit(dataset)
	actual_dataset = imp.transform(dataset)
	pd.testing.assert_frame_equal(expected_dataset, actual_dataset, rtol=0.24, atol=0.24)

def test_mixed_data_imputer():
        data = [[0.3243, np.nan, "stretch", np.nan],
                [np.nan, 0.9823, np.nan, "adult"],
                [0.3724, 0.9820, "dip", "child"],
                [np.nan, 0.9825, np.nan, "child"],
                [0.3592, 0.9824, "dip", "adult"],
                [np.nan, np.nan, "stretch", "child"],
                [0.3350, 0.9828, "stretch", np.nan],
                ]
        dataset = pd.DataFrame(data, columns=["col1", "col2", "act", "age"])

        answer_data = [[0.3243, 0.9823, "stretch", "child"],
                       [0.3607, 0.9823, "dip", "adult"],
                       [0.3724, 0.9820, "dip", "child"],
                       [0.3475, 0.9825, "stretch", "child"],
                       [0.3592, 0.9824, "dip", "adult"],
                       [0.3475, 0.9823, "stretch", "child"],
                       [0.3350, 0.9828, "stretch", "child"],
                       ]

        expected_dataset = pd.DataFrame(answer_data, columns=["col1", "col2", "act", "age"])

        imp = MissForestImputer()
        actual_dataset = imp.fit_transform(dataset)
        pd.testing.assert_frame_equal(expected_dataset, actual_dataset, rtol=0.24, atol=0.24)

@pytest.fixture
def sample_dataframe():
		data = {
			'Name': ['Alice', 'Bob', 'Charlie'],  # Categorical
			'Age': [25, 30, 35],  # Numerical
			'Gender': ['Female', 'Male', 'Male']  # Categorical
		}
		df = pd.DataFrame(data)
		return df

def test__get_categorical_features(sample_dataframe):
		expected = ['Name', 'Gender']
		imp = MissForestImputer()
		imp.fit(sample_dataframe)
		actual = imp.get_categorical_features()
		assert set(actual) == set(expected), "The categorical columns were not correctly identified."

def test__get_numerical_features(sample_dataframe):
		expected = ['Age']
		imp = MissForestImputer()
		imp.fit(sample_dataframe)
		actual = imp.get_numerical_features()
		assert set(actual) == set(expected), "The numerical columns were not correctly identified."

dataframes_to_test_categorical = [
    pd.DataFrame({
        'Category1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'Category2': ['D', 'E', 'F', 'D', np.nan, 'F', 'D', 'E', np.nan, 'D', 'E', 'F', np.nan, 'E', 'F'],
        'Category3': ['G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I'],
        'Category4': [np.nan, 'K', 'L', 'J', np.nan, 'L', 'J', 'K', np.nan, 'J', np.nan, 'L', 'J', 'K', np.nan],
        'Category5': [np.nan] * 10 + ['O', 'M', 'N', 'O', 'M'],
    }),
	
    pd.DataFrame({
        'Cat1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'Cat2': ['D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F'],
        'Cat3': ['G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I'],
        'Cat4': ['J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L'],
        'Cat5_with_missing': ['M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', np.nan],
    }),
	
    pd.DataFrame({
        'Cat1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'Cat2': ['D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F'],
        'Cat3': ['G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I'],
        'Cat4': ['J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L'],
        'Cat5': ['M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', 'O', 'M', 'N', "O"],
    }),
	
    pd.DataFrame({
		'Cat1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'Cat2': ['D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F'],
        'Cat3': ['G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I'],
        'Cat4': ['J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L', 'J', 'K', 'L'],
        'Cat5_with_90_missing': [np.nan] * 13 + ['N', 'O'],  # 90% missing
    }),
	
    pd.DataFrame({
		'Num1': range(1, 16),  # Numerical data
        'Cat1_with_30_missing': ['A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'B', 'C', 'A', 'B', 'C'],  # Categorical with missing
        'Num2': range(16, 31),  # Numerical data
        'Cat2_with_30_missing': [np.nan, 'E', 'F', np.nan, 'E', 'F', np.nan, 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F'],  # Categorical with missing
        'Num3': range(31, 46),  # Numerical data
    })
]

@pytest.mark.parametrize("sample_dataset_categorical", dataframes_to_test_categorical)
def test_fit_function_on_categorical_data(sample_dataset_categorical):
        # Initialize your DataImputer and fit it to the deterministic dataset
        imputer = MissForestImputer()
        imputer.fit(sample_dataset_categorical)

         # Filter the dataset to only include categorical columns with at least one missing value
        categorical_cols = sample_dataset_categorical.select_dtypes(include=['object', 'category']).columns
        categorical_with_missing = sample_dataset_categorical[categorical_cols].dropna(axis=1, how='all')
    

        # Assert the n_features_in_ attribute
        assert imputer.iterative_categorical_imputer.n_features_in_ == categorical_with_missing.shape[1], "n_features_in_ does not match."

        # Assert the feature_names_in_ attribute
        assert np.array_equal(imputer.iterative_categorical_imputer.feature_names_in_, categorical_with_missing.columns), "feature_names_in_ does not match."

        # Assert the n_features_with_missing_ attribute
        expected_missing_features_count = categorical_with_missing.isnull().any().sum()
        assert imputer.iterative_categorical_imputer.n_features_with_missing_ == expected_missing_features_count, "n_features_with_missing_ does not match."

@pytest.mark.parametrize("sample_dataset_categorical", dataframes_to_test_categorical)
def test_transform_function_on_categorical_data(sample_dataset_categorical):
    # Initialize your DataImputer and fit it to the dataset
    imputer = MissForestImputer()
    imputer.fit(sample_dataset_categorical)

    # Record the original data and types for comparison
    original_data = sample_dataset_categorical.copy()
    original_dtypes = sample_dataset_categorical.dtypes

    # Transform the dataset
    transformed_data = imputer.transform(sample_dataset_categorical)

    # Verify no missing values
    assert not transformed_data.isnull().any().any(), "Transformed dataset should have no missing values."

    # Verify unchanged non-missing data
    for col in original_data.columns:
        unchanged_mask = original_data[col].notna()
        assert all(original_data[col][unchanged_mask] == transformed_data[col][unchanged_mask]), f"Data in column {col} should remain unchanged."

    # Verify data types stayed the same
    assert all(transformed_data.dtypes == original_dtypes), "Data types should remain unchanged after transformation."

@pytest.mark.parametrize("sample_dataset_categorical", dataframes_to_test_categorical)
def test_fit_transform_function_on_categorical_data(sample_dataset_categorical):
    # Initialize your DataImputer and fit it to the dataset
    imputer = MissForestImputer()
    
    # Record the original data and types for comparison
    original_data = sample_dataset_categorical.copy()
    original_dtypes = sample_dataset_categorical.dtypes

    # Transform the dataset
    transformed_data = imputer.fit_transform(sample_dataset_categorical)

    # Verify no missing values
    assert not transformed_data.isnull().any().any(), "Transformed dataset should have no missing values."

    # Verify unchanged non-missing data
    for col in original_data.columns:
        unchanged_mask = original_data[col].notna()
        assert all(original_data[col][unchanged_mask] == transformed_data[col][unchanged_mask]), f"Data in column {col} should remain unchanged."

    # Verify data types stayed the same
    assert all(transformed_data.dtypes == original_dtypes), "Data types should remain unchanged after transformation."


datasets_to_test_numerical = [
	pd.DataFrame({
		'Num1': range(1, 16),  # Continuous numerical values
		'Num2': np.concatenate([range(100, 105), np.array([np.nan] * 5), range(105, 110)]),  # Introduce NaN values in the middle
		'Num3': range(21, 36),  # Another set of continuous numerical values
		'Num4': np.concatenate([np.array([np.nan]), range(200, 204), np.array([np.nan] * 5), range(204, 209)]),  # Mixed NaN positions
		'Num5': np.concatenate([np.array([np.nan] * 10), range(300, 305)]),  # 10 NaN values followed by numbers
	}),
	pd.DataFrame({
    'Column1': range(1, 16),
    'Column2': range(16, 31),
    'Column3': range(31, 46),
    'Column4': range(46, 61),
    'Column5': range(61, 76),
	}),
	pd.DataFrame({
    'Column1': range(1, 15 + 1),
    'Column2': [np.nan] * 13 + list(range(1, 15 - 13 + 1)),
    'Column3': range(101, 101 + 15),
    'Column4': [np.nan] * 13 + list(range(11, 11 + 15 - 13)),
    'Column5': range(201, 201 + 15),
	}),
	pd.DataFrame({
    'Cat1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Cat2': ['D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F', 'D', 'E', 'F'],
    'Cat3': ['G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I', 'G', 'H', 'I'],
    'Num1': [1, 2, 3, np.nan, 5, 6, 7, np.nan, 9, 10, 11, 12, np.nan, 14, 15],
    'Num2': [101, 102, np.nan, 104, 105, 106, np.nan, 108, 109, 110, np.nan, 112, 113, 114, 115],
	})	
]

@pytest.mark.parametrize("sample_dataset_numerical", datasets_to_test_numerical)
def test_fit_function_on_numerical_data(sample_dataset_numerical):
    # Initialize your DataImputer and fit it to the deterministic dataset
    imputer = MissForestImputer()
    imputer.fit(sample_dataset_numerical)

    # Filter the dataset to only include numerical columns with at least one missing value
    numerical_cols = sample_dataset_numerical.select_dtypes(include=[np.number]).columns
    numerical_with_missing = sample_dataset_numerical[numerical_cols].dropna(axis=1, how='all')

    # Assert the n_features_in_ attribute for numerical data handling
    assert imputer.iterative_numerical_imputer.n_features_in_ == numerical_with_missing.shape[1], "n_features_in_ does not match for numerical data."

    # Assert the feature_names_in_ attribute for numerical data handling
    assert np.array_equal(imputer.iterative_numerical_imputer.feature_names_in_, numerical_with_missing.columns), "feature_names_in_ does not match for numerical data."

    # Assert the n_features_with_missing_ attribute for numerical data handling
    expected_missing_features_count = numerical_with_missing.isnull().any().sum()
    assert imputer.iterative_numerical_imputer.n_features_with_missing_ == expected_missing_features_count, "n_features_with_missing_ does not match for numerical data."

@pytest.mark.parametrize("sample_dataset_numerical", datasets_to_test_numerical)
def test_transform_function_on_numerical_data(sample_dataset_numerical):
    # Initialize your DataImputer and fit it to the deterministic dataset
    imputer = MissForestImputer()
    imputer.fit(sample_dataset_numerical)

    # Record the original data and types for comparison
    original_data = sample_dataset_numerical.copy()
    original_dtypes = sample_dataset_numerical.dtypes

    # Transform the dataset
    transformed_data = imputer.transform(sample_dataset_numerical)

    # Verify no missing values
    assert not transformed_data.isnull().any().any(), "Transformed dataset should have no missing values."

    # Verify unchanged non-missing data
    for col in original_data.columns:
        unchanged_mask = original_data[col].notna()
        assert all(original_data[col][unchanged_mask] == transformed_data[col][unchanged_mask]), f"Data in column {col} should remain unchanged."
    print(transformed_data.dtypes, original_dtypes)
    # Verify data types stayed the same
    assert all(transformed_data.dtypes == original_dtypes), "Data types should remain unchanged after transformation."

@pytest.mark.parametrize("sample_dataset_numerical", datasets_to_test_numerical)
def test_fit_transform_function_on_numerical_data(sample_dataset_numerical):
    # Initialize your DataImputer and fit it to the dataset
    imputer = MissForestImputer()
    
    # Record the original data and types for comparison
    original_data = sample_dataset_numerical.copy()
    original_dtypes = sample_dataset_numerical.dtypes

    # Transform the dataset
    transformed_data = imputer.fit_transform(sample_dataset_numerical)

    # Verify no missing values
    assert not transformed_data.isnull().any().any(), "Transformed dataset should have no missing values."

    # Verify unchanged non-missing data
    for col in original_data.columns:
        unchanged_mask = original_data[col].notna()
        assert all(original_data[col][unchanged_mask] == transformed_data[col][unchanged_mask]), f"Data in column {col} should remain unchanged."
	
    print(transformed_data.dtypes, original_dtypes)
    # Verify data types stayed the same
    assert all(transformed_data.dtypes == original_dtypes), "Data types should remain unchanged after transformation."
