from EOTE import EOTE
import pytest
from unittest.mock import MagicMock
from unittest.mock import Mock
import pandas as pd
import numpy as np
from EOTE.Ditectors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from EOTE.Utils import DataFrameBuilderAff

def test_mixed_dataset(capsys):
    director = EOTEDirector(EoteWithMissForestInTerminalBuilder())

    # Building EOTE
    eote = director.get_eote()

    train_dataset = DataFrameBuilderAff().create_dataframe(
        "EOTE_test/Datasets/datasets_arff/mixed_training.arff")

    test_dataset = DataFrameBuilderAff().create_dataframe(
        "EOTE_test/Datasets/datasets_arff/mixed_testing.arff")
    
    X_train = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
    y_train = train_dataset.iloc[:, train_dataset.shape[1] - 1:train_dataset.shape[1]]

    X_test = test_dataset.iloc[:, 0:test_dataset.shape[1] - 1]
    y_test = test_dataset.iloc[:, test_dataset.shape[1] - 1: test_dataset.shape[1]]

    eote.train(X_train, y_train)

    y_pred = eote.classify(X_test)
    print(y_pred)
    
    assert eote._EOTE__is_trained == True
    for value in y_pred:
         assert pd.api.types.is_numeric_dtype(value[0]), "Series should contain only numerical values"
   

    eote.classify_and_interpret(X_test.loc[1])
    assert eote._EOTE__is_trained == True
    captured = capsys.readouterr()
    assert captured.out, "The function should have printed something, but it didn't."



@pytest.fixture
def setup_class():
    director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
    # Instance of the class containing __preprocess_dataset
    obj = EOTE()
    obj.class_verification_method = Mock()
    obj.attribute_remover = Mock()
    obj.imputer = Mock()
    obj.encoder = Mock()
    
    # Setup return values and side effects
    obj.class_verification_method.verify_if_valid.return_value = True
    obj.attribute_remover.remove_invalid_attributes.side_effect = lambda x: x  # Assuming it returns DataFrame as is
    obj.imputer.fit_transform.side_effect = lambda x: x.fillna(0)  # Simple fill NA for testing
    obj.encoder.fit.side_effect = lambda x: None
    obj.encoder.transform.side_effect = [pd.DataFrame(np.array([[1, 0], [0, 1], [1, 0]]), columns=['encoded_feature1', 'encoded_feature2'])]
    obj.encoder.get_feature_names_out.side_effect = lambda: ['encoded_feature1', 'encoded_feature2']
    return obj

def test_preprocess_dataset(setup_class):
    # Creating a DataFrame with missing values and nominal data
    x = pd.DataFrame({
        'A': [1, 2, None],
        'B': ['red', 'blue', 'red']
    })
    y = pd.Series([0, 1, 1])

    # Run the preprocessing
    setup_class._EOTE__preprocess_dataset(x, y)

    # Verify methods were called
    setup_class.class_verification_method.verify_if_valid.assert_called_once_with(y)
    setup_class.attribute_remover.remove_invalid_attributes.assert_called_once()
    setup_class.imputer.fit_transform.assert_called_once()
    setup_class.encoder.fit.assert_called_once()
    setup_class.encoder.transform.assert_called_once()
    assert 'encoded_feature1' in setup_class._EOTE__x_encoded.columns
    assert 'encoded_feature2' in setup_class._EOTE__x_encoded.columns
    assert not setup_class._EOTE__x_encoded.isnull().any().any()  # Check no missing values
    #Assert that no original categorical columns are present in the final dataset
    assert not any(col in setup_class._EOTE__x_encoded.columns for col in x['B']), \
        "Categorical columns should not be present after encoding"



@pytest.fixture
def setup_obj():
    obj = EOTE()
    obj._EOTE__x_encoded = pd.DataFrame({
        'feature1': [10, 20, 30, 40],
        'feature2_a': [0, 1, 0, 1],
        'feature2_b': [1, 0, 1, 0],
    })
    obj.x = pd.DataFrame({
        'feature1': [10, 20, 30, 40],
        'feature2': ['red', 'blue', None, 'blue'],
    })
    return obj

def test_get_feature_x(setup_obj):
    feature_mock = Mock()
    feature_mock.name.return_value = 'feature2'
    feature_mock.missing_value_objects = [2]

    result = setup_obj._EOTE__get_feature_x(feature_mock)

    # Assertions
    expected = pd.DataFrame({
        'feature1': [10, 20, 40],
    })
    assert len(expected.values) == len(result.values), "Lists differ in size"
    for item1, item2 in zip(expected.values, result.values):
        assert item1 == item2, f"Items differ: {item1} != {item2}"

def test_get_feature_y(setup_obj):
    feature_mock = Mock()
    feature_mock.name.return_value = 'feature2'
    feature_mock.missing_value_objects = [2]

    result = setup_obj._EOTE__get_feature_y(feature_mock)

    # Assertions
    expected = pd.DataFrame({
        'feature1': ['red', 'blue', 'blue'],
    })
    assert len(expected.values) == len(result.values), "Lists differ in size"
    for item1, item2 in zip(expected.values, result.values):
        assert item1 == item2, f"Items differ: {item1} != {item2}"


@pytest.fixture
def setup_object():
    obj = EOTE()
    obj._EOTE__original_feature_names = ['feature1', 'feature2', 'feature3']
    feature_mock1 = Mock()
    feature_mock1.name.return_value = 'feature1'
    feature_mock1.type.return_value = 'Nominal'
    feature_mock1.values.return_value = ['A', 'B', 'C']

    feature_mock2 = Mock()
    feature_mock2.name.return_value = 'feature2'
    feature_mock2.type.return_value = 'Numeric'  # Numeric should not be affected

    feature_mock3 = Mock()
    feature_mock3.name.return_value = 'feature3'
    feature_mock3.type.return_value = 'Nominal'
    feature_mock3.values.return_value = ['X', 'Y']

    obj.features = [feature_mock1, feature_mock2, feature_mock3]
    return obj

def test_replace_unknown_values(setup_object):
    df = pd.DataFrame({
        'feature1': ['A', 'D', 'E'],  # 'D' and 'E' are unknown
        'feature2': [1, 2, 3],       # Should remain unaffected
        'feature3': ['X', 'Z', 'Y']  # 'Z' is unknown
    })

    result = setup_object._EOTE__replace_unknown_values(df)
    
    # Check the correct handling of nominal and non-nominal features
    assert result['feature1'].isna().sum() == 2, "Feature1 should have 2 NaNs for unknown values"
    assert result['feature2'].equals(df['feature2']), "Feature2 should remain unchanged"
    assert result['feature3'].isna().sum() == 1, "Feature3 should have 1 NaN for unknown values"
    assert pd.isna(result.at[1, 'feature3']), "Feature3 should have NaN at index 1"
