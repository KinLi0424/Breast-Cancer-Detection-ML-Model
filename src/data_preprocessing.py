import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(test_size=0.2, random_state=42):
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

# Example usage:
# X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
