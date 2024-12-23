# model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier model.
    Args:
        X_train (array-like): The features for training.
        y_train (array-like): The target variable.
    Returns:
        model: The trained model.
    """
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using the test data.
    Args:
        model: The trained model.
        X_test (array-like): The features for testing.
        y_test (array-like): The true labels.
    Returns:
        None
    """
    # Predict on test data
    y_pred = model.predict(X_test)

    print(f"Accuracy: {model.score(X_test, y_test):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, scaler, model_path, scaler_path):
    """
    Save the trained model and scaler using joblib.
    Args:
        model: The trained model.
        scaler: The feature scaler.
        model_path (str): Path to save the model.
        scaler_path (str): Path to save the scaler.
    Returns:
        None
    """
    # Save trained model
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler have been saved.")

def load_model(model_path, scaler_path):
    """
    Load the trained model and scaler from disk.
    Args:
        model_path (str): Path to the saved model.
        scaler_path (str): Path to the saved scaler.
    Returns:
        model, scaler: The loaded model and scaler.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
