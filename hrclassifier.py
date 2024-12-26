import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Input, Concatenate
# from tf.keras.layers import Dense, Dropout,Input, Concatenate, 
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import argparse
import os
import numpy as np

def load_data(files):

    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
    # Define features and targets
    feature_columns = ["head", "neck", "left_ear", "right_ear", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
    # [f"person{i}{x}" for i in range(1, 5) for x in 'abcdefghij']
    # target_columns = [f"Person{i}_hand" for i in range(1, 5)]
    
    def parse_feature(value):
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            # Extract the first number from the tuple-like string
            # x,y = float(value.split(",")[0].strip("()"))
            x,y = value.strip("()").split(",")
            return float(x),float(y)
        else: 
            return float(value), float(0)  # If already a float
    
    # X = data[feature_columns]
    # X = data[feature_columns].apply(lambda x: float(x.split(",")[0].strip("()")) if isinstance(x, str) else x)
    # X = data[feature_columns].applymap(parse_feature)
    X_x = pd.DataFrame()
    X_y = pd.DataFrame()
    for feature in feature_columns:
        X_x[feature], X_y[feature] = zip(*data[feature].map(parse_feature))
    yt = data['hand_raised']

    # Scale the features 
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

    # enc_y1 
    encoder = LabelEncoder()
    ey =    encoder.fit_transform(yt).reshape(-1,1)
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
    print("X_x:", X_x)
    print("X_y:", X_y)
    # print("y:", yc, yc.shape, type(yc))

    return X_x,X_y, ey

def build_model(input_dim):

    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output layer for 4 outputs
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Suitable for binary labels (hand raised or not)
                  metrics=['accuracy', 'mae', 'binary_accuracy', 'binary_crossentropy','mse'])

    return model

def build_model_2_inputs(input_dim):
   
    input_x = Input(shape=(input_dim,), name="Input_X")
    input_y = Input(shape=(input_dim,), name="Input_Y")

    print("in_x:", input_x)
    print("in_y:", input_y)

    # Process x-coordinates
    x_branch = Dense(64, activation='relu')(input_x)
    x_branch = Dropout(0.3)(x_branch)

    # Process y-coordinates
    y_branch = Dense(64, activation='relu')(input_y)
    y_branch = Dropout(0.3)(y_branch)

    # Combine x and y branches
    combined = Concatenate()([x_branch, y_branch])

    # Fully connected layers after combining
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid', name="Output")(x)  # Binary output

    # Define the model
    model = Model(inputs=[input_x, input_y], outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mae', 'binary_accuracy', 'binary_crossentropy', 'mse'])
    return model

def main(folder_path, train_files, val_files, epochs, batch_size):

    # Resolve full paths for training and validation files
    train_file_paths = [os.path.join(folder_path, file) for file in train_files]
    val_file_paths = [os.path.join(folder_path, file) for file in val_files]

    # Load training and validation data
    # X_train, y_train = load_data(train_file_paths)
    # X_val, y_val = load_data(val_file_paths)
    X_train_x, X_train_y, y_train = load_data(train_file_paths)
    X_val_x, X_val_y, y_val = load_data(val_file_paths)

    #random shuffle 
    X_sh_x,X_sh_y, y_sh = shuffle(X_train_x, X_train_y,y_train)
    print("shuffle:", X_sh_x, " ", X_sh_y)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Build the model
    model = build_model_2_inputs(input_dim=10)#X_train.shape[1])

    # Train the model
    history = model.fit(
        # X_train, y_train,
        [X_sh_x, X_sh_y],y_sh,
        validation_data=([X_val_x,X_val_y], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    print("loss:", history.history['loss'])
    if len(history.history['loss']) < epochs:
        print("Training stopped early due to early stopping.")

    # Save the model
    model.save("classifier_2in_E0.h5")
    print("Model saved as 'keypoint_model.h5'.")

    print("Training and Validation Results:")
    for key in history.history.keys():
        print(f"{key}: {history.history[key][-1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on keypoint dataset.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing dataset files.")
    parser.add_argument('--train_files', nargs='+', required=True, help="List of training file names.")
    parser.add_argument('--val_files', nargs='+', required=True, help="List of validation file names.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")

    args = parser.parse_args()

    main(args.folder_path, args.train_files, args.val_files, args.epochs, args.batch_size)
