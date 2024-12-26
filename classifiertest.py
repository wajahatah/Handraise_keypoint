
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import argparse

# Function to preprocess the test data
def preprocess_data(file):
    data = pd.read_csv(file)

    # Define features and target columns
    feature_columns = ["head", "neck", "left_ear", "right_ear", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
    target_column = 'hand_raise'

    def parse_feature(value):
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            return float(value.split(",")[0].strip("()"))
        return float(value)

    X = data[feature_columns].applymap(parse_feature)
    y = data[target_column] if target_column in data.columns else None

    # # Scale the features
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

    if y is not None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    return X, y

def main(test_file, model_file):

    # Preprocess the test data
    X_test, y_test = preprocess_data(test_file)

    # Load the trained model
    model = load_model(model_file)

    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    if y_test is not None:
        accuracy = accuracy_score(y_test, predicted_classes)
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No target column in test data. Only predictions are available.")

    print("Predictions:", predicted_classes)

    # if y_test is not None:
    #     loss, accuracy, mae, binary_accuracy, binary_crossentropy, mse = model.evaluate(X_test, y_test, verbose=0)
    #     print("Test Results:")
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Mean Absolute Error (MAE): {mae:.4f}")
    #     print(f"Binary Accuracy: {binary_accuracy:.4f}")
    #     print(f"Binary Crossentropy: {binary_crossentropy:.4f}")
    #     print(f"Mean Squared Error (MSE): {mse:.4f}")
    # else:
    #     print("No target column in test data. Running predictions only.")
    #     predictions = model.predict(X_test)
    #     print("Predictions:", predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained neural network on a dataset.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument('--model_file', type=str, required=True, help="Path to the trained model file.")

    args = parser.parse_args()

    main(args.test_file, args.model_file)