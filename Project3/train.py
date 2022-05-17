import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_dir = "text-dnn-data"  # Local directory to store data
blobstore_datadir = data_dir  # Blob store directory to store data in
target_column_name = "y"
feature_column_name = "X"


data_dir = "text-dnn-data"  # Local directory to store data
blobstore_datadir = data_dir  # Blob store directory to store data in
target_column_name = "y"
feature_column_name = "X"


def get_20newsgroups_data():
    """Fetches 20 Newsgroups data from scikit-learn
    Returns them in form of pandas dataframes
    """
    remove = ("headers", "footers", "quotes")
    categories = [
        "rec.sport.baseball",
        "rec.sport.hockey",
        "comp.graphics",
        "sci.space",
    ]

    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )
    
    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )


    return data_train, data_test


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--batch_size', type=int, default=100, help="Number of images in each mini-batch")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Batch size:", np.int(args.batch_size))
    run.log("Learning rate:", np.int(args.learning_rate))

    data_train, data_test = get_20newsgroups_data()
    # converting text data into vectors of numerical values using tf-idf to form feature vector
    vectorizer = TfidfVectorizer()
    data_train_vectors = vectorizer.fit_transform(data_train.data)
    data_test_vectors = vectorizer.transform(data_test.data)
    
    # store training feature matrix in "Xtr"
    x_train = data_train_vectors

    # store training response vector in "ytr"
    y_train = data_train.target
    
    # store testing feature matrix in "Xtt"
    x_test = data_test_vectors

    # store testing response vector in "ytt"
    y_test = data_test.target
    
    model = OneVsRestClassifier(LinearSVC(C=args.C, max_iter=args.max_iter)).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename='outputs/text_dnn_sklearn_model.pkl')

if __name__ == '__main__':
    main()
