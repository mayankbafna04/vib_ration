#main.py

import os
from data_preprocessing import load_and_preprocess_data
from model import train_model, evaluate_model, save_model, load_model
from util import check_file_exists

def main():
    dataset_path = 'archive/processed_file.csv'
    if not check_file_exists(dataset_path):
        print(f"Error: The dataset file {dataset_path} does not exist.")
        return
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(dataset_path)

    #training
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    #save 
    model_path = '/Users/apple/Desktop/Vibration/models/best_random_forest_model.pkl'
    scaler_path = '/Users/apple/Desktop/Vibration/models/scaler.pkl'
    save_model(model, scaler, model_path, scaler_path)

    if check_file_exists(model_path) and check_file_exists(scaler_path):
        print(f"Model and scaler are saved at {model_path} and {scaler_path}.")

    loaded_model, loaded_scaler = load_model(model_path, scaler_path)
    print("Model and scaler have been loaded for inference.")

    #new data for prediction 
    new_data = [[-828,0.843871845796248,-1.191999934267655,-0.901472945422427,-0.9035858371506359,-1.3030074862951824,-0.6456541072996341,-0.9768282127958325,-0.6225729015807281]]  # Replace with actual new data

    #prediction
    new_data_scaled = loaded_scaler.transform(new_data)
    prediction = loaded_model.predict(new_data_scaled)
    print(f"Prediction for new data: {prediction}")

if __name__ == "__main__":
    main()
