#Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import geopy.distance
import time
import joblib

#load classifier 
loaded_clf = joblib.load('clf_model')

#load encoder
loaded_le = joblib.load('label_encoder')

#function for calculating angles
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360 - angle 
    
    return angle

def mp4video_app(video, clf = loaded_clf, le = loaded_le):
    
    # MediaPipe Functions Declaration
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pred = None
    counter = None
    exercise = None

    #pull-ups statements initialization
    pull_up_state = 'up'
    pull_ups_counter = 0

    #squats statements initialization
    squats_state = 'down'
    squats_counter = 0

    #plank statements initialization
    plank_sec = 0
    plank_start = None
    
    cap = cv2.VideoCapture(video)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        start = time.time()
        while cap.isOpened():
            
            ret, frame = cap.read()

            #Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Temp lists update 
                temp_list = [landmark.x for landmark in landmarks]
                temp_list_y = [landmark.y for landmark in landmarks]
                temp_list.extend(temp_list_y)

                inpt = ([temp_list])
                encoded_pred = clf.predict(inpt)
                pred = le.inverse_transform(encoded_pred)
                prob = clf.predict_proba(inpt)

                if pred == 'Pull-Ups':

                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    pull_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                    if pull_up_state == 'up' and pull_angle < 90:
                        pull_ups_counter+=1
                        pull_up_state = 'down'
                    elif pull_up_state == 'down' and pull_angle > 120:
                        pull_up_state = 'up'

                if pred == 'Squats':

                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    squat_angle = calculate_angle(l_hip, l_knee, l_ankle)

                    if squats_state == 'up' and squat_angle > 150:
                        squats_counter += 1
                        squats_state = 'down'
                    elif squats_state == 'down' and squat_angle < 100:
                        squats_state='up'

                if pred == 'Plank':
                    if plank_start == None:
                        plank_start = time.time()
                    else:
                        current = time.time()
                        plank_sec += current - plank_start
                        plank_start = current

            except:
                pass

            # Render predictor

            if pred == 'Pull-Ups':
                exercise = 'Pull-Ups'
                counter = pull_ups_counter
            elif pred == 'Squats':
                exercise = 'Squats'
                counter = squats_counter
            elif pred == 'Plank':
                exercise = 'Plank'
                counter = round(plank_sec, 3)
            elif pred == 'Rest':
                exercise = 'Rest'
                counter = '-'

            # Setup status box
            cv2.rectangle(image, (0,0), (350,200), (65,100,75), -1)#(input, start_point, end_point, color, thickness)

            # View Prediction
            cv2.putText(image, 'REPS : '+str(counter),
                            (10,160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(image, str(exercise), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                         )

            # Window name 
            cv2.imshow('Video Feed', image)

            # Break key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # End of Video

    cap.release()
    cv2.destroyAllWindows()
