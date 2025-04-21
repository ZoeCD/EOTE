import sys
sys.path.append(".")
import pytest
from EOTE.Utils.Data_interpretation.terminal_output_formatter import TerminalOutputFormatter

@pytest.fixture
def sample_anomaly_rules():
    return ["If (Length ≤ 0.42249999940395355) AND (Viscera_weight ≤ 0.17899999767541885) then (Sex = M)",
            "If (Shucked_weight ≤ 0.47200000286102295) then (0.12 ≤ Whole_weight ≤ 1.2865)",
            "If (Whole_weight ≤ 1.112749993801117) then (0.0415 ≤ Shucked_weight ≤ 0.5195)"]

@pytest.fixture
def sample_normal_rules():
    return ["Classifier has only one node, results are: (0.31 ≤ Length ≤ 0.78)",
            "If (Whole_weight ≤ 1.1260000467300415) then (0.024 ≤ Viscera_weight ≤ 0.2965)"]

@pytest.fixture
def sample_feature_names():
    return ["Length", "Whole_weight", "Viscera_weight", "Shucked_weight", "Sex"]

def test_terminal_output_formatter_anomaly_case(sample_anomaly_rules, sample_normal_rules, sample_feature_names, capsys):
    instance = [0.21, 0.0385, 0.0155, 0.0085, 'M']
    final_score = -0.1
    terminal_output_formatter = TerminalOutputFormatter()
    result = terminal_output_formatter.produce_output(instance, final_score, sample_anomaly_rules, sample_normal_rules, sample_feature_names)
    assert "Anomaly" == result

    captured = capsys.readouterr()
    assert captured.out != "" 

def test_terminal_output_formatter_normal_case(sample_anomaly_rules, sample_normal_rules, sample_feature_names, capsys):
    instance = [0.21, 0.0385, 0.0155, 0.0085, 'M']
    final_score = 0.3
    terminal_output_formatter = TerminalOutputFormatter()
    result = terminal_output_formatter.produce_output(instance, final_score, sample_anomaly_rules, sample_normal_rules, sample_feature_names)

    assert "Normal" == result
    captured = capsys.readouterr()
    assert captured.out != "" 

    