from ultralytics import YOLO
import os
import cv2
import csv
import numpy as np
import pandas as pd
import ast
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import argparse

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

person_ranges = {
    "Person1": lambda x: 90 < x < 367,
    "Person2": lambda x: 370 <= x <= 730,
    "Person3": lambda x: 740 <= x <= 1080,
    "Person4": lambda x: x > 1080
}

frame_width = 1280
frame_height = 780

# Subtraction values for normalization
subtraction_values = {
    "Person2": 370,
    "Person3": 740,
    "Person4": 1065
}

# Keypoint labels
keypoint_labels = ["head", "neck", "left_ear", "right_ear", "left_shoulder",
                   "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]

output_dir = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output"
# frames_output_dir = os.path.join(output_dir, "v29")
# os.makedirs(frames_output_dir, exist_ok=True)

def keypoint_model(frame,model_file):
    model = YOLO(model_file)
    # video = video_path

    # output_csv_path = os.path.join(output_dir, 'v29s.csv')
    # with open(output_csv_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)

    #     header = ["Frame"] + [f"{person}_{kp}" for person in person_ranges.keys() for kp in keypoint_labels]
    #     csv_writer.writerow(header)

        # Check if the video capture opened successfully
        # if not cap.isOpened():
        #     print("Error: Could not open video.")
        #     exit()

        # frame_count = 0  # To track frame number

        # Read and process the video frame by frame
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break  # Exit the loop if no more frames are available

        # Run inference on the current frame
    frame = cv2.resize(frame, (1280, 720))
    results = model(frame)

    # frame_keypoints = {person: ["(0,0,0.0)"] * len(keypoint_labels) for person in person_ranges.keys()}

    # keypoints_dict = {person: {} for person in person_ranges.keys()}
    keypoints_dict = {person: {kp_idx: {'x': 0, 'y': 0} for kp_idx in range(len(keypoint_labels))} for person in person_ranges.keys()}

    # Iterate over each detected object and process their keypoints
    for result in results:
        keypoints = result.keypoints  # Access the keypoints object

        if keypoints is not None:
            # Get the data attribute, which contains x, y, and confidence values
            keypoints_data = keypoints.data
            for person_keypoints in keypoints_data:
                for kp_idx, keypoint in enumerate(person_keypoints[:len(keypoint_labels)]):
                    x, y = keypoint[0].item(), keypoint[1].item()

                    # Determine which person the keypoint belongs to based on x-coordinate ranges
                    for person, condition in person_ranges.items():
                        if condition(x):
                                keypoints_dict[person][kp_idx] = {'x': round(x, 2), 'y': round(y, 2)}
                            # if kp_idx not in keypoints_dict[person]:
                            #     keypoints_dict[person][kp_idx] = []
                            # keypoints_dict[person][kp_idx].append({
                            #     'x': round(x,2),
                            #     'y': round(y,2)
                            # })
                    #         frame_keypoints[person][kp_idx] = f"({x:.2f},{y:.2f},{confidence:.2f})"
                    #         break

                    # kp for person in person_ranges.keys() for kp in frame_keypoints[person]

                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        # cv2.putText(
                        #     frame,
                        #     f"({int(x)}, {int(y)})",
                        #     (int(x) + 5, int(y) - 5),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 0, 0),
                        #     1
                        # )

    # Write the processed frame keypoints to the CSV
    # row = [frame_count] + [kp for person in person_ranges.keys() for kp in frame_keypoints[person]]
    # csv_writer.writerow(row)

    # frame_count += 1

    # cap.release()
    # cv2.destroyAllWindows()
    print("keypoints", keypoints_dict)
    return keypoints_dict

# def parse_and_normalize(value, person, subtraction_values):
def parse_and_normalize(keypoint, person, subtraction_values):
    try:
        # Handle missing points or zero entries
        # if value == 0 or value == "0":
        # if value == "(0.00,0.00)":
        #     return (0, 0)
        
        # Convert string representation to a tuple
        # x, y, confidence = ast.literal_eval(value)
        x, y = keypoint['x'], keypoint['y']

        
        # # Handle invalid or missing data
        if x == 0 or y == 0:# or confidence == 0:
            return (0, 0)
        
        # Apply subtraction for specific persons
        # subtraction = subtraction_values.get(person, 0)
        # x -= subtraction
        # y -= subtraction
        
        # Normalize the coordinates
        x_norm = x / frame_width
        y_norm = y / frame_height
        
        return (x_norm, y_norm)
    except (ValueError, SyntaxError, TypeError):
        return (0, 0)

def transform(keypoints_dict):

    normalized_keypoints_dict = {person: {} for person in keypoints_dict.keys()}
    
    for person, keypoints in keypoints_dict.items():
        for kp_idx, keypoint in keypoints.items():
            if keypoint['x'] == 0 and keypoint['y'] == 0:
                normalized_keypoints_dict[person][kp_idx] = (0, 0)  # Default for missing keypoints
            else:
                normalized_keypoints_dict[person][kp_idx] = parse_and_normalize(keypoint, person, subtraction_values)
        
        # for kp_idx, keypoint_list in keypoints.items():
        #     normalized_keypoint_list = []
        #     for keypoint in keypoint_list:
        #         # x, y, confidence = keypoint['x'], keypoint['y'], keypoint['confidence']
        #         x, y = keypoint[0].item(), keypoint[1].item()
        #         # normalized_x, normalized_y= parse_and_normalize(keypoint, person)
        #         normalized_keypoints = parse_and_normalize(keypoint, person,subtraction_values)
        #         normalized_keypoint_list.append(normalized_keypoints)
        #             # 'x': round(normalized_x, 3),
        #             # 'y': round(normalized_y, 3)
        #             # 'confidence': round(normalized_confidence, 3)
        #         # })
        #     normalized_keypoints_dict[person][kp_idx] = normalized_keypoint_list
    # print("personNorm:",person)
    return normalized_keypoints_dict,person


    """
    input_csv=in_path
    output_csv=output_dir

    df = pd.read_csv(input_csv)

# Prepare transformed data
    stacked_data = []

    for person in ["Person1", "Person2", "Person3", "Person4"]:
        # Extract relevant columns for this person
        # person_columns = [f"{person}{chr(i)}" for i in range(ord('a'), ord('a') + len(body_parts))]
        person_columns = [f"{person}_{part}" for part in keypoint_labels]
        person_data = df[person_columns]
        
        # Normalize data for each column
        parsed_data = person_data.applymap(lambda value: parse_and_normalize(value, person, subtraction_values))
        
        # Prepare formatted data for this person
        formatted_data = pd.DataFrame({
            "frame": df["Frame"], #df["Frame Name"].str.extract(r'(\d+)$').astype(int).iloc[:, 0],
            "person": person,
            **{
                # part: parsed_data[col].apply(lambda t: f"({t[0]:.3f},{t[1]:.3f})")
                # for part, col in zip(body_parts, person_columns)
                part: parsed_data[f"{person}_{part}"].apply(lambda t: f"({t[0]:.3f},{t[1]:.3f})")
                for part in keypoint_labels
            }
        })
        
        stacked_data.append(formatted_data)

    # Combine all transformed person data
    result = pd.concat(stacked_data, ignore_index=True)

    # Save the result to a new CSV file
    output_file = result.to_csv(output_csv, index=False)
    return output_file
"""
"""
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

    # scaler = StandardScaler()
    # # Scale the features
    # X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

    if y is not None:
        encoder = LabelEncoder()

        y = encoder.fit_transform(y)

    return X, y"""

def main(video_path): #, keypoint_model_weights,classifier):

    # video_path = folder_path
    cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     exit()

    # model = load_model(classifier)
    model = load_model("C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/classifier_2in_30E.h5")
    X_test_x = []
    X_test_y = []
    person_number = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(1280,720))
        # keypoints = keypoint_model(frame, keypoint_model_weights)
        keypoints = keypoint_model(frame,"runs/pose/trail4/weights/best_y11.pt" )
        # print("keypoints:", keypoints)
        for person, kp_idct in keypoints.items():
            for kp_id, coords in kp_idct.items():
                x,y = int(coords['x']), int(coords['y'])
                cv2.circle(frame,(x,y), 5, (0,255,0), -1)


        # points
        normalized_keypoints,person = transform(keypoints)
        print("normalized", normalized_keypoints)

        # for person_keypoints in normalized_keypoints.values():
        #     for kp in person_keypoints.values():
        #         X_test.append(kp)

        # p = normalized_keypoints.keys
        # print("person", p)
        """
        # Single input
        X_test = [kp for person in normalized_keypoints.values() for kp_list in person.values() for kp in kp_list]
        print("kplist", X_test)
        if not X_test:
            print("No keypoints detected, unable to make predictions.")
            return

        # Convert to NumPy array for model input
        # X_test = np.array(X_test).reshape(-1, len(keypoint_labels) * 2)
        X_test = np.array(X_test).reshape(-1, 10)
        
        
        if len(X_test) == 0:
            print("No keypoints detected, skipping predictions.")
        else:
            predictions = model.predict(X_test)
            print("Predictions:", predictions)
        
        
        for i, predictions in enumerate(predictions):
            if predictions[0] == 1: #is True:
                cv2.putText(frame, "hand_raise", (50+i*30 , 200),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),1)
                # cv2.putText(frame, str(person), (100+i*30 , 200),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),1)

        print("kp", X_test)
        # X_test = np.array(X_test)
        """

        X_test_x = [kp_list[0] for person in normalized_keypoints.values() for kp_list in person.values()]# for kp in kp_list]
        X_test_y = [kp_list[1] for person in normalized_keypoints.values() for kp_list in person.values()]# for kp in kp_list]
        # X_test_y = [kp for person in normalized_keypoints.values() for kp_list in person.values() for kp in kp_list]

        # for person_id, person_keypoints in normalized_keypoints.items():
        #     print(f"Person ID: {person_id}, Keypoints: {person_keypoints}")

        #     x_coords = [kp[0] for kp_list in person_keypoints.values() for kp in kp_list]
        #     y_coords = [kp[1] for kp_list in person_keypoints.values() for kp in kp_list]
        #     print ("coords:",x_coords, y_coords)
        # #     # Only consider valid persons with sufficient keypoints
        #     if len(x_coords) > 0 and len(y_coords) > 0:
        #         X_test_x.append(x_coords)
        #         X_test_y.append(y_coords)
        #         person_number.append(person_id)

        print("x test:", X_test_x, X_test_y)
        # if not X_test_x or not X_test_y:
            # print("No keypoints detected, unable to make predictions.")
            # continue

        # Convert to NumPy arrays for model input
        X_test_x = np.array(X_test_x).reshape(-1, 10)  # Reshape based on x-coordinates
        X_test_y = np.array(X_test_y).reshape(-1, 10)  # Reshape based on y-coordinates

        # if len(X_test_x) == 0 or len(X_test_y) == 0:
        #     print("No keypoints detected, skipping predictions.")
        # else:
            # Predict using the model
        predictions = model.predict([X_test_x, X_test_y])
        print("Predictions:", predictions)
        # predictions = predictions.reshape(-1)
        # paired_predictions = zip(predictions[::2], predictions[1::2])
        # paired_predictions = [(predictions[i], predictions[i + 1]) for i in range(0, len(predictions), 2)]



    # Annotate predictions on the frame
        for i, prediction in enumerate(predictions):
            # for i, (pred1,pred2) in enumerate(paired_predictions):
            if prediction[0] > 0.6:  # Assuming a threshold for binary classification
                # print("person num:", person_number[i])
            # if pred1 > 0.5 and pred2 > 0.5:
                # person_label = f"Person{person_number[i]}: Hand Raised"
                cv2.putText(frame, "Hand Raised", (180 + i * 280, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    # cv2.putText(frame, person_label, (50 + i * 30, 200),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    
    # predictions = model.predict(X_test)
    # print("Predictions:", predictions)
    # Preprocess the test data
    # X_test = preprocess_data(points)

    # Load the trained model

    # predicted_classes = (predictions > 0.5).astype(int).flatten()

    # if y_test is not None:
            # Delay of 50 milliseconds to slow down the video playback
    #     accuracy = accuracy_score(y_test, predicted_classes)
    #     print(f"Accuracy: {accuracy:.4f}")
    # else:
    #     print("No target column in test data. Only predictions are available.")

    # print("Predictions:", predicted_classes)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained neural network on a dataset.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the test data CSV file.")
    # parser.add_argument('--video', nargs='+', required=True, help="Path to the test data CSV file.")
    # parser.add_argument('--keypoint_model_weights', type=str, required=True, help="Path to the trained model file.")
    # parser.add_argument('--classifier', type=str, required=True, help="Path to the trained model file.")

    args = parser.parse_args()

    main(args.folder_path)
    #, args.keypoint_model_weights, args.classifier)
    # main(args.folder_path, args.video, args.keypoint_model_weights, args.classifier)