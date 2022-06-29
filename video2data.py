#Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import glob

# MediaPipe Functions Declaration
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video Feed 

def video_to_data(video_name, exercise, csv_name):

    # Intialialize dataframe where joints coordinates are stored
    df = pd.DataFrame(columns = [landmark.name for landmark in mp_pose.PoseLandmark])

    # x storage
    df_x = pd.DataFrame(columns = [landmark.name for landmark in mp_pose.PoseLandmark])

    # y storage
    df_y = pd.DataFrame(columns = [landmark.name for landmark in mp_pose.PoseLandmark])

    # Initialize annotation list 
    label_list = []

    # Run mp4 video 
    cap = cv2.VideoCapture(video_name)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame =cap.read()
            if ret:
                cv2.imshow('Video', frame)

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
                    temp_list = [landmark for landmark in landmarks]
                    temp_list_x = [landmark.x for landmark in landmarks]
                    temp_list_y = [landmark.y for landmark in landmarks]

                    # Dataframes update 
                    df.loc[len(df)] = temp_list
                    df_x.loc[len(df)] = temp_list_x
                    df_y.loc[len(df)] = temp_list_y

                except:
                    pass
                
                #update exercise list
                label_list.append(exercise)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                                         )

                # Window name 
                cv2.imshow('Annotation Feed', image)

                #Break key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

        # Close window 
        cap.release()
        cv2.destroyAllWindows()
        
        #connvert labels list to dataframe 
        df_labels = pd.DataFrame (label_list, columns = ['LABEL'])
    
        #combine the three dataframes to one 
        final_df = pd.concat([df_labels, df_x, df_y], axis=1, join='inner')
        
        #df to csv 
        final_df.to_csv(r'data/_'+str(csv_name)+'_.csv', index=False)
        
        #return csv file with landmark features for the video


def run_file(file_name, label):
    
    filenames = glob.glob(str(file_name)+"/*.mp4")
    counter = 0
    
    for video in filenames:
        counter+=1
        video_to_data(video, label,label+str(counter))
        
    #runs "video_to_data" for every mp4 in a train file
