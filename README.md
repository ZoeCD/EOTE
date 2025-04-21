# EOTE

An Explainable Outlier Tree-base autoEncoder detector. This outlier detector classifier is able to work with tabular data (numerical and categorical) and output a rule-based explanation of the prediction.

This is a python implementation using scikit-learn decision trees. 

For categorical trees, we created a new python implementation of the DTAE algorithm [1]. This version is faster than the C# implementation at https://github.com/miguelmedinaperez/DTAE.

When working with missing data, this implementation performs data imputation. In this case, the output rules must be read with care.

# References
[1] D. L. Aguilar, M. A. Medina-Pérez, O. Loyola-González, K.-K. Raymond Choo, E. Bucheli-Susarrey, "Towards an interpretable autoencoder: A decision tree-based autoencoder and its application in anomaly detection," <i>IEEE Transactions on Dependable and Secure Computing</i>, doi: 10.1109/TDSC.2022.3148331.