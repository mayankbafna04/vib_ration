# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Load the dataset, preprocess it, and split it into train and test sets.
    Args:
        file_path (str): The path to the dataset CSV file.
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data split into training and testing sets.
    """

    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.drop(columns=['fault'])  # Features (exclude the target column)
    y = df['fault']  # Labels (target column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
