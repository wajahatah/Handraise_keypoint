from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import load_model
import joblib  # For loading .pkl files
import pickle  # Alternative for loading .pkl files

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

person_ranges = {
    "Person1": lambda x: x < 367,
    "Person2": lambda x: 370 <= x <= 730,
    "Person3": lambda x: 740 <= x <= 1080,
    "Person4": lambda x: x > 1080
}

frame_width = 1280
frame_height = 780

subtraction_values = {
    "Person2": 370,
    "Person3": 740,
    "Person4": 1065
}

keypoint_labels = ["head", "neck", "left_ear", "right_ear", "left_shoulder",
                   "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]


# def load_random_forest_model(model_path):
#     if model_path.endswith(".h5"):
#         print(f"Loading model from {model_path} (H5 format)...")
#         return load_model(model_path)
#     elif model_path.endswith(".pkl"):
#         print(f"Loading model from {model_path} (PKL format)...")
#         with open(model_path, 'rb') as file:
#             return pickle.load(file)
#     else:
#         raise ValueError("Unsupported file format. Please use .h5 or .pkl files.")
def load_random_forest_model(model_path):
    if model_path.endswith(".h5"):
        print(f"Loading H5 model from {model_path}...")
        return load_model(model_path)  # Neural network models
    elif model_path.endswith(".pkl"):
        print(f"Loading PKL model from {model_path}...")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            if not hasattr(model, 'predict'):
                raise TypeError(f"The loaded object from {model_path} is not a model.")
            return model
    else:
        raise ValueError("Unsupported file format. Use .h5 or .pkl.")


def keypoint_model(frame, model_file):
    model = YOLO(model_file)
    frame = cv2.resize(frame, (1280, 720))
    results = model(frame)

    keypoints_dict = {
        person: {kp_idx: {'x': 0, 'y': 0} for kp_idx in range(len(keypoint_labels))}
        for person in person_ranges.keys()
    }

    for result in results:
        keypoints = result.keypoints
        if keypoints is not None:
            keypoints_data = keypoints.data
            for person_keypoints in keypoints_data:
                for kp_idx, keypoint in enumerate(person_keypoints[:len(keypoint_labels)]):
                    x, y = keypoint[0].item(), keypoint[1].item()
                    for person, condition in person_ranges.items():
                        if condition(x):
                            keypoints_dict[person][kp_idx] = {'x': round(x, 2), 'y': round(y, 2)}
    return keypoints_dict


def transform(keypoints_dict):
    normalized_keypoints_dict = {person: {} for person in keypoints_dict.keys()}
    for person, keypoints in keypoints_dict.items():
        for kp_idx, keypoint in keypoints.items():
            normalized_keypoints_dict[person][kp_idx] = parse_and_normalize(keypoint, person, subtraction_values)
    return normalized_keypoints_dict


def parse_and_normalize(keypoint, person, subtraction_values):
    x, y = keypoint['x'], keypoint['y']
    if x == 0 or y == 0:
        return (0, 0)
    subtraction = subtraction_values.get(person, 0)
    x -= subtraction
    x_norm = x / frame_width
    y_norm = y / frame_height
    return (x_norm, y_norm)


def main(folder_path, keypoint_model_weights, classifier_path):
    video_path = folder_path
    cap = cv2.VideoCapture(video_path)
    classifier = load_random_forest_model(classifier_path)
    X_test = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = keypoint_model(frame, keypoint_model_weights)
        normalized_keypoints = transform(keypoints)
        X_test = [kp for person in normalized_keypoints.values() for kp in person.values()]

        if X_test:
            X_test = np.array(X_test).reshape(-1, len(keypoint_labels) * 2)
            predictions = classifier.predict(X_test)
            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    cv2.putText(frame, "hand_raise", (50 + i * 30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained neural network on a dataset.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the video file.")
    parser.add_argument('--keypoint_model_weights', type=str, required=True, help="Path to YOLO model weights.")
    parser.add_argument('--classifier', type=str, required=True, help="Path to the Random Forest model (.h5 or .pkl).")
    args = parser.parse_args()
    main(args.folder_path, args.keypoint_model_weights, args.classifier)
