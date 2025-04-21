import sys
sys.path.append(".")
import pytest
from EOTE.Utils.Data_interpretation import TxtFileOutputFormatter
import os


def test_txt_output_formatter():
    # Setup
    expected_file_path = 'report.txt'
    formatter = TxtFileOutputFormatter(expected_file_path)
    
    # Ensure any existing file is removed before the test (for a clean slate)
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)
    
    instance = [0.21, 0.0385, 0.0155, 0.0085, 'M']
    final_score = -0.1
    # Action
    formatter.produce_output(instance, final_score,
                       anomaly_rules = ["Rule 1 = Anomaly", "Rule 2 = Anomaly"],
                       normal_rules = ["Rule 1 = Normal", "Rule 2 = Normal"],
                       feature_names = ["Length", "Whole_weight", "Viscera_weight", "Shucked_weight", "Sex"])
    
    # Assertion
    assert os.path.isfile(expected_file_path), "Report file was not created at the expected location."
    
    # Cleanup
    os.remove(expected_file_path)  # Clean up the file after the test

def test_txt_output_anomaly():
    # Setup
    expected_file_path = 'report.txt'
    formatter = TxtFileOutputFormatter(expected_file_path)
    
    # Ensure any existing file is removed before the test (for a clean slate)
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)
    
    instance = [0.21, 0.0385, 0.0155, 0.0085, 'M']
    final_score = 1
    # Action
    formatter.produce_output(instance, final_score,
                       anomaly_rules = ["Rule 1 = Anomaly", "Rule 2 = Anomaly"],
                       normal_rules = ["Rule 1 = Normal", "Rule 2 = Normal"],
                       feature_names = ["Length", "Whole_weight", "Viscera_weight", "Shucked_weight", "Sex"])
    
    # Assertion
    assert os.path.isfile(expected_file_path), "Report file was not created at the expected location."
    
    # Cleanup
    os.remove(expected_file_path)  # Clean up the file after the test