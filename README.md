# EOTE: Explainable Outlier Tree-based AutoEncoder

**EOTE** is an interpretable anomaly detection system for tabular data that provides rule-based explanations for its predictions. Unlike black-box models, EOTE generates human-readable decision rules explaining *why* an instance is classified as normal or anomalous.

## Key Features

- **Interpretable by Design**: Generates decision tree-based rules explaining each prediction
- **Handles Mixed Data**: Works seamlessly with both numerical and categorical features
- **Missing Data Support**: Built-in MissForest imputation for handling incomplete datasets
- **Semi-Supervised Learning**: Trains on normal class data to detect anomalies

## How It Works

EOTE uses a novel **per-feature autoencoding** approach:

1. **For each feature** in the dataset, trains a decision tree to predict that feature from all other features
2. **Calculates anomaly scores** by comparing predicted vs. actual values across all features
3. **Extracts decision paths** from the trees to generate interpretable rules
4. **Combines scores** to produce a final classification with supporting explanations

This approach provides both accurate anomaly detection and transparent, actionable insights into *why* instances are flagged as anomalous.

## Quick Start

### Basic Usage

```python
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from EOTE.Utils import DataFrameBuilderAff
import pandas as pd

# 1. Build EOTE using Builder pattern
director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
eote = director.get_eote()

# 2. Load and prepare data
train_data = pd.read_csv("normal_data.csv")
X_train = train_data.iloc[:, :-1]  # Features
y_train = train_data.iloc[:, -1]   # Class labels (should be single class)

# 3. Train the model
eote.train(X_train, y_train)

# 4. Classify new instances
test_data = pd.read_csv("test_data.csv")
X_test = test_data.iloc[:, :-1]

# Get anomaly scores
scores = eote.classify(X_test)
print(f"Anomaly scores: {scores}")

# 5. Get interpretable explanation for a specific instance
eote.classify_and_interpret(X_test.loc[0])
```

### Example Output

When you call `classify_and_interpret()`, EOTE provides output like:

```
Instance: [age=45, income=120000, status=unemployed, education=16]

Classification: Anomaly (score: 0.82)

Anomaly Rules:
  - If (age > 40) AND (income > 100000) then (status = unemployed)
    [Expected: employed, but predicted unemployed with high confidence]

Normal Rules:
  - If (education ≤ 18) then (credit_score = 680)
    [Matches expected pattern for this education level]
```

### Output Options

EOTE supports two output formats:

#### Terminal Output (colored, interactive)
```python
from EOTE.Builders import EoteWithMissForestInTerminalBuilder

director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
eote = director.get_eote()
# Output will be printed to console with colors
```

#### File Output (for logging/archiving)
```python
from EOTE.Builders import EoteWithMissForestInTxTFileBuilder

director = EOTEDirector(EoteWithMissForestInTxTFileBuilder("results.txt"))
eote = director.get_eote()
# Output will be written to results.txt
```

## Advanced Usage

### Working with ARFF Files

EOTE includes utilities for loading ARFF (Attribute-Relation File Format) datasets:

```python
from EOTE.Utils import DataFrameBuilderAff

# Load ARFF file
dataset = DataFrameBuilderAff().create_dataframe("data/mydata.arff")
```

### Custom Configuration

You can create custom builders to configure EOTE differently:

```python
from EOTE import EOTE
from EOTE.Protocols import EOTEBuilder
from sklearn.preprocessing import OneHotEncoder
from EOTE.Utils import MissForestImputer, TerminalOutputFormatter

class CustomEOTEBuilder(EOTEBuilder):
    def __init__(self):
        self.eote = EOTE()

    def set_data_imputer(self):
        # Use custom imputation settings
        self.eote.imputer = MissForestImputer(max_iter=30, tol=0.1)
        return self

    # Implement other required builder methods...

    def build(self):
        return self.eote
```

### Hyperparameter Tuning

The default builders use these hyperparameters:

- **CCP Alphas for tree pruning**: `[0.025, 0.010, 0.005]`
- **Minimum categorical values**: 2 unique values with minimum 3 instances each
- **MissForest**: `max_iter=20`, `tol=0.24`

These can be customized through custom builder implementations.

## Understanding the Results

### Anomaly Scores

- **Score > 0**: Instance classified as **Anomaly**
- **Higher positive scores**: Stronger anomaly signal

### Interpreting Rules

Decision rules show the path through decision trees that led to the prediction:

- **Anomaly Rules**: Features that contribute to the anomaly classification
- **Normal Rules**: Features that support normal behavior

Each rule follows the format:
```
If (condition1) AND (condition2) then (predicted_feature = value)
```

### Important Note on Missing Data

When working with datasets containing missing values, EOTE performs automatic imputation using the MissForest algorithm. **The decision rules may reference imputed values** rather than original data. Interpret these rules with awareness that some values may have been inferred.

## Architecture

EOTE follows clean software engineering principles:

- **Builder Pattern**: Flexible configuration of EOTE instances
- **Director Pattern**: Ensures proper construction sequence
- **Strategy Pattern**: Interchangeable scorers, imputers, and formatters

### Project Structure

```
EOTE/
├── EOTE.py                 # Main EOTE class
├── Builders/               # Builder pattern implementations
├── Directors/              # Director pattern for orchestration
├── Trees/                  # Feature tree implementations
├── Utils/                  # Data processing, interpretation, utilities
└── Protocols/              # Interface definitions
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest EOTE_test/

# Run with coverage
pytest EOTE_test/ --cov=EOTE --cov-report=html

# Run specific test file
pytest EOTE_test/DTAE_test.py -v
```

## Examples

See the `example_terminal_output.py` and `example_txt_output.py` files for complete working examples.

## Related Work

EOTE is based on the DTAE (Decision Tree-based AutoEncoder) algorithm. For the original C# implementation, see: https://github.com/miguelmedinaperez/DTAE

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citations

If you use EOTE in your research, please cite:

### Primary Reference (Latest):
```bibtex
@article{caballero2026explainable,
  title={An explainable autoencoder integrating regression and classification trees for anomaly detection},
  author={Caballero-Dominguez, Z. and Monroy, R. and Medina-P{\'e}rez, M.~A.},
  journal={Expert Systems with Applications},
  volume={296},
  pages={128975},
  year={2026},
  doi = {10.1016/j.eswa.2025.128975},
  publisher={Elsevier}
}
```

### Original DTAE Algorithm:
```bibtex
@article{aguilar2022interpretable,
  title={Towards an interpretable autoencoder: A decision tree-based autoencoder and its application in anomaly detection},
  author={Aguilar, D. L. and Medina-P{\'e}rez, M. A. and Loyola-Gonz{\'a}lez, O. and Choo, K.-K. Raymond and Bucheli-Susarrey, E.},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2022},
  doi={10.1109/TDSC.2022.3148331},
  publisher={IEEE}
}
```

---

