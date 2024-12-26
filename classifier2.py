import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load multiple CSV files for training and testing
# train_files = glob.glob("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/handraisedataset/v1.csv")  # Update path to your training files
# test_files = glob.glob("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/handraisedataset/v1.csv")    # Update path to your testing files

df = pd.read_csv("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/handraisedataset/v1.csv")

# Combine all training files into one DataFrame
# df_train = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)
# df_test = pd.concat([pd.read_csv(file) for file in test_files], ignore_index=True)

# Separate features and target
X = df.drop(columns=['frame', 'person', 'hand_raised'])
y = df['hand_raised']
# X_train = df_train.drop(columns=['frame', 'person', 'hand_raised'])
# y_train = df_train['hand_raised']
# X_test = df_test.drop(columns=['frame', 'person', 'hand_raised'])
# y_test = df_test['hand_raised']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build the model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
