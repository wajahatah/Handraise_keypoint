import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os
import numpy as np
import joblib
import glob
import h5py

def load_data(files):

    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
    # Define features and targets
    feature_columns = ["head", "neck", "left_ear", "right_ear", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
    # [f"person{i}{x}" for i in range(1, 5) for x in 'abcdefghij']
    # target_columns = [f"Person{i}_hand" for i in range(1, 5)]
    
    def parse_feature(value):
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            # Extract the first number from the tuple-like string
            return float(value.split(",")[0].strip("()"))
        return float(value)  # If already a float
    
    # X = data[feature_columns]
    # X = data[feature_columns].apply(lambda x: float(x.split(",")[0].strip("()")) if isinstance(x, str) else x)
    X = data[feature_columns].applymap(parse_feature)
    y = data['hand_raised']

    # Scale the features 
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

    encoder = LabelEncoder()
    # enc_y1 
    ey =    encoder.fit_transform(y).reshape(-1,1)
    # # enc_y1 = encoder.transform(y)#
    # encoder = LabelEncoder()
    # enc_y2 = encoder.fit_transform(y.iloc[:,1]).reshape(-1,1)
    # # enc_y2 = encoder.transform(y)#
    # encoder = LabelEncoder()
    # enc_y3 = encoder.fit_transform(y.iloc[:,2]).reshape(-1,1)
    # # # enc_y3 = encoder.transform(y)#
    # encoder = LabelEncoder()
    # enc_y4 = encoder.fit_transform(y.iloc[:,3]).reshape(-1,1)
    # enc_y4 = encoder.transform(y)#
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # enc_y = encoder.transform(y)#.reshape(-1,1)
    # yc = np.concatenate([enc_y1,enc_y2,enc_y3,enc_y4], axis=1)
    print("X:", X)
    # print("y:", yc, yc.shape, type(yc))

    return X, ey

# def build_model(input_dim):
#     """
#     Build a simple neural network model.

#     Args:
#         input_dim (int): Number of input features.

#     Returns:
#         model (Sequential): Compiled Keras model.
#     """
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=input_dim),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')  # Output layer for 4 outputs
#     ])

#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',  # Suitable for binary labels (hand raised or not)
#                   metrics=['accuracy', 'mae', 'binary_accuracy', 'binary_crossentropy','mse'])

#     return model

def main(folder_path, train_files, val_files, nest, rs):

    # Resolve full paths for training and validation files
    train_file_paths = [os.path.join(folder_path, file) for file in train_files]
    val_file_paths = [os.path.join(folder_path, file) for file in val_files]

    # Load training and validation data
    X_train, y_train = load_data(train_file_paths)
    X_val, y_val = load_data(val_file_paths)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    rf_model = RandomForestClassifier(n_estimators=nest, random_state=rs)
    
    rf_model.fit(X_train,y_train)

    y_pred = rf_model.predict(X_val)

    accuracy = accuracy_score(y_val,y_pred)
    print("accuracy: ", accuracy)
    print("report", classification_report(y_val,y_pred))


    joblib.dump(rf_model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")

    # with h5py.File('random_forest_model.h5', 'w') as h5file:
    #     model_bytes = joblib.dumps(rf_model)
    #     h5file.create_dataset('random_forest_model', data=np.void(model_bytes))
    # print("Random Forest model saved as 'random_forest_model.h5'")

    # Build the model
    # model = build_model(input_dim=10)#X_train.shape[1])

    # # Train the model
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=epochs,
    #     batch_size=batch_size
    #     # callbacks=[early_stopping]
    # )

    # print("loss:", history.history['loss'])
    # if len(history.history['loss']) < epochs:
    #     print("Training stopped early due to early stopping.")

    # # Save the model
    # model.save("classifier50e.h5")
    # print("Model saved as 'keypoint_model.h5'.")

    # print("Training and Validation Results:")
    # for key in history.history.keys():
    #     print(f"{key}: {history.history[key][-1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on keypoint dataset.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing dataset files.")
    parser.add_argument('--train_files', nargs='+', required=True, help="List of training file names.")
    parser.add_argument('--val_files', nargs='+', required=True, help="List of validation file names.")
    parser.add_argument('--nest', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--rs', type=int, default=42, help="Batch size for training.")

    args = parser.parse_args()

    main(args.folder_path, args.train_files, args.val_files, args.nest, args.rs)
