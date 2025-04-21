import sys
sys.path.append(".")
import pytest
import numpy as np
from EOTE.Utils import AnomalyDomainScorer, AnomalyBoxplotScorer
import pandas as pd

# Example data for testing
data_array = np.array([1, 2, 3, 4, 5])
data_frame = pd.DataFrame(data_array, columns=['feature'])

@pytest.fixture
def domain_scorer():
    scorer = AnomalyDomainScorer()
    scorer.set_domain(data_array)
    return scorer

@pytest.fixture
def boxplot_scorer():
    scorer = AnomalyBoxplotScorer()
    scorer.set_domain(data_frame['feature'])
    return scorer

# Tests for AnomalyDomainScorer
def test_domain_scorer_set_domain(domain_scorer):
    assert domain_scorer.domain == (1, 5), "Domain should be set to the minimum and maximum of the data."

def test_domain_scorer_calculate_score_normal(domain_scorer):
    assert domain_scorer.calculate_score_normal(3, 3) == 1.0, "Score should be 1.0 for no difference."
    assert domain_scorer.calculate_score_normal(1, 5) == 0.0, "Score should be 0.0 for max difference."

def test_domain_scorer_calculate_score_anomaly(domain_scorer):
    assert domain_scorer.calculate_score_anomaly(3, 3) == 0, "Score should be 0 for no difference."
    assert domain_scorer.calculate_score_anomaly(1, 5) == 1.0, "Score should be 1.0 for max difference."

# Tests for AnomalyBoxplotScorer
def test_boxplot_scorer_set_domain(boxplot_scorer):
    q1, q3 = np.quantile(data_array, [0.25, 0.75])
    iqr = q3 - q1
    expected_domain = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    assert boxplot_scorer.domain == expected_domain, "Domain should be set to the boxplot fences."

def test_boxplot_scorer_calculate_score_normal(boxplot_scorer):
    normal_value, predicted_value = 3, 3
    score = boxplot_scorer.calculate_score_normal(normal_value, predicted_value)
    assert score == 1.0, "Score should be 1.0 for no difference."

def test_boxplot_scorer_calculate_score_anomaly(boxplot_scorer):
    normal_value, predicted_value = 1, 5
    score = boxplot_scorer.calculate_score_anomaly(normal_value, predicted_value)
    # The exact value depends on the domain calculation
    assert 0 < score <= 1.0, "Score should be between 0 and 1.0 for some difference."
