from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

def dist(a,b,c,d):
    return math.sqrt((c-a)**2 + (d-b)**2)

def calculate_angle(point1, point2, point3):

    point1 = point1.cpu().numpy() if isinstance(point1, torch.Tensor) else point1
    point2 = point2.cpu().numpy() if isinstance(point2, torch.Tensor) else point2
    point3 = point3.cpu().numpy() if isinstance(point3, torch.Tensor) else point3

    # Convert points to vectors
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

if __name__ == "__main__":

    model = YOLO("runs/pose/trail32/weights/best_yv8.pt")
    # video_path = "FP_HR_01.mp4"
    video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_19_05.mp4"
    # video_path = "Cam_19_10.mp4"

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Run inference on the current frame
        results = model(frame)

        # Iterate over each detected person and print their keypoints
        for result in results:
            keypoints = result.keypoints  # Get keypoints as a numpy array or tensor
            if keypoints is not None: 
                keypoints_data=keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    for kp in person_keypoints:
                        x, y, confidence = kp
                        if confidence > 0.5:  # Optional: Only draw keypoints with sufficient confidence
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 3)  # Draw the keypoint
                    
                    ls = person_keypoints[4]  # Keypoint 4
                    le = person_keypoints[5]  # Keypoint 5
                    lw = person_keypoints[6]  # Keypoint 6
                    rs = person_keypoints[7]  # Keypoint 7
                    re = person_keypoints[8]  # Keypoint 8
                    rw = person_keypoints[9]  # Keypoint 9

                    print("riGHt shoulder", rs)

                    lsx =ls[0].item() 
                    lsy =ls[1].item()
                    lex =le[0].item()
                    ley =le[1].item()
                    lwx =lw[0].item()
                    lwy =lw[1].item()
                    rsx =rs[0].item() 
                    rsy =rs[1].item() 
                    rex =re[0].item() 
                    rey =re[1].item()
                    rwx =rw[0].item()
                    rwy =rw[1].item()

                    print("lsx", rsx, "lsy", rsy)

                    cv2.circle(frame, (int(lex), int(ley)), 5, (254, 32, 32), -1)  # Keypoint D
                    cv2.circle(frame, (int(lwx), int(lwy)), 5, (254, 32, 32), -1)  # Keypoint D
                    cv2.circle(frame, (int(lsx), int(lsy)), 9, (254, 32, 32), 3)  # Keypoint D
                    cv2.circle(frame, (int(rex), int(rey)), 5, (254, 32, 32), -1)  # Keypoint D
                    cv2.circle(frame, (int(rwx), int(rwy)), 5, (254, 32, 32), -1)  # Keypoint D
                    cv2.circle(frame, (int(rsx), int(rsy)), 9, (254, 32, 32), 3)  # Keypoint D


                    if all(value != 0 for value in [lsx,lsy,lex,ley,lwx,lwy,rsx,rsy,rex,rey,rwx,rwy]):
                        ld =dist(lsx,lsy,lwx,lwy)

                        rd =dist(rsx,rsy,rwx,rwy)

                        print("ld", ld)
                        print("rd", rd)

                        cv2.putText(frame, str(ld), (int(lex), int(ley)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                        cv2.putText(frame, str(rd), (int(rex), int(rey)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)


                        cv2.line(frame, (int(lex), int(ley)), (int(lwx), int(lwy)), (255,255,255), 1)
                        cv2.line(frame, (int(lsx), int(lsy)), (int(lwx), int(lwy)), (255,255,255), 1)
                        cv2.line(frame, (int(rex), int(rey)), (int(rwx), int(rwy)), (255,255,255), 1)
                        cv2.line(frame, (int(rsx), int(rsy)), (int(rwx), int(rwy)), (255,255,255), 1)


                        # cv2.line(frame, (lex,ley),(lwx,lwy),(235,248,0),2)
                        # cv2.line(frame, (rex,rey),(rwx,rwy),(235,248,0),2)

                        # cv2.putText(frame, str(lsy), (int(lsx), int(lsy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),6)
                        # cv2.putText(frame, str(lwy), (int(lwx), int(lwy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),6)
                        # cv2.putText(frame, str(rsy), (int(rsx), int(rsy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),6)
                        # cv2.putText(frame, str(rwy), (int(rwx), int(rwy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),6)

                        langle = calculate_angle(ls,le,lw)
                        rangle = calculate_angle(rs,re,rw)

                        print("left angle:", langle)
                        print("right angle:",rangle)

                        #  and 
                        if langle > 130:
                            if lwy < lsy:
                                # and 420 < ld > 350:
                                cv2.putText(frame, "Hand raise", (int(lsx), int(lsy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                        #  and 
                        if rangle > 130:
                           if rwy < rsy: 
                        #    and rd > 350:
                               cv2.putText(frame, "Hand raise", (int(rsx), int(rsy)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)

            # for result in results:
            #     annotated_frame = result.plot()  # Draw keypoints and bounding boxes on the frame
            pp = cv2.resize(frame, (1280, 720))
            cv2.imshow('Pose Detection', pp)

            # Press 'q' to quit the video display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close display window
    cap.release()
    cv2.destroyAllWindows()
