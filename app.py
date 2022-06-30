#importing libraries
import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import time
import joblib
import numpy as np
from PIL import Image

#load classifier and encoder
loaded_clf = joblib.load('appmods/clf_model')
loaded_enc = joblib.load('appmods/label_encoder')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


#mediapipe function declaration
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#title
st.title('AI Trainer')
st.text('Built with OpenCV and MediaPipe')

#sidebar
st.sidebar.title('Lite Bar')

sidebar_im1 = Image.open('appmods/sidebar1.jpg')
st.sidebar.image(sidebar_im1)

#selectbox
app_mode = st.sidebar.selectbox(
    'Select Mode', ['Demo', 'Your Video', 'About'])

if app_mode == 'About':

    st.title('About The Project')
    st.write('We will put some text here')

elif app_mode == 'Demo' or app_mode == 'Your Video':

    tpfile = tempfile.NamedTemporaryFile(delete=False)

    if app_mode == 'Demo':
        pulldemo = st.sidebar.button('Pull-Ups Demo')
        squatsdemo = st.sidebar.button('Squats Demo')
        plankdemo = st.sidebar.button('Plank Demo')
        kpi1, kpi2, kpi3 = st.sidebar.columns(3)
        video_name = 'appmods/pulldemo.mp4'
        if pulldemo:
            video_name = 'appmods/pulldemo.mp4'
        elif squatsdemo:
            video_name = 'appmods/squatsdemo.mp4'
        elif plankdemo:
            video_name = 'appmods/plankdemo.mp4'

    else:
        video_file_buffer = st.sidebar.file_uploader(
            'Upload Your Video', type='mp4')
        use_webcam = st.sidebar.button('Use Webcam')
        kpi1, kpi2, kpi3 = st.sidebar.columns(3)

        if video_file_buffer is not None:
            tpfile.write(video_file_buffer.read())
            video_name = tpfile.name

        if use_webcam:
            video_name = 0

    stframe = st.empty()
    vid = cv2.VideoCapture(video_name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Exercise**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Reps**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**FPS**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    pred = None
    counter = None
    exercise = None
    fps = 0

    #pull-ups statements initialization
    pull_up_state = 'up'
    pull_ups_counter = 0

    #squats statements initialization
    squats_state = 'down'
    squats_counter = 0

    #plank statements initialization
    plank_sec = 0
    plank_start = None

    #classifier - encoder load
    clf = loaded_clf
    le = loaded_enc

    # Setup mediapipe instance
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        prevTime = 0

        while vid.isOpened():

            ret, frame = vid.read()

            # Make detection
            results = pose.process(frame)
            frame.flags.writeable = True

            #Extract landmarks
            try:
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
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    pull_angle = calculate_angle(
                        l_shoulder, l_elbow, l_wrist)

                    if pull_up_state == 'up' and pull_angle < 90:
                        pull_ups_counter += 1
                        pull_up_state = 'down'
                    elif pull_up_state == 'down' and pull_angle > 120:
                        pull_up_state = 'up'

                if pred == 'Squats':
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    squat_angle = calculate_angle(l_hip, l_knee, l_ankle)

                    if squats_state == 'up' and squat_angle > 150:
                        squats_counter += 1
                        squats_state = 'down'
                    elif squats_state == 'down' and squat_angle < 100:
                        squats_state = 'up'

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

            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                                color=(0, 0, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                                color=(255, 255, 255), thickness=2, circle_radius=2)
                                      )

            #FPS counter
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime

            #Dashboard
            kpi1_text.write(
                        f"<h1 style='text-align: center; color:red;'>{str(exercise)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(
                        f"<h1 style='text-align: center; color:red;'>{str(counter)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(
                        f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, height=540)
            stframe.image(frame, channels='BGR', use_column_width=True)
