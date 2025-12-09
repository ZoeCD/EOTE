import sys
sys.path.append(".")
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from EOTE.Utils import DataFrameBuilderAff



def main():
    # Create the director by passing the DTAE builder
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

    # Classify and interpret example
    eote.classify_and_interpret(X_test.loc[1])


if __name__ == '__main__':
    main()
