import csv
import copy
import itertools
import cv2
import mediapipe as mp
import numpy as np
import mouse
from utils import CvFpsCalc
from model import KeyPointClassifier

import threading
from queue import Queue

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

points = []
count = 0
h = None
camIdx = 0
screen_width = 1920
screen_height = 1080
#making two different functions, one for gesture recognition and the other for control.
def keypoint_landmak_identification(input_queue, output_queue):
    print("Thread started")
    keypoint_classifier = KeyPointClassifier()
    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    while True:
        
        data = input_queue.get()
        debug_image = data["debug_image"]
        multi_hand_landmarks = data["multi_hand_landmarks"]
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                # Landmark calculation
                landmark_list = calc_landmark_list(
                    debug_image, hand_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(
                    pre_processed_landmark_list)
                print("Output queued")
                output_queue.put(hand_sign_id)


def main():
    ###Multi threading related code
    input_queue,output_queue = Queue(),Queue()
    thread = threading.Thread(target=keypoint_landmak_identification,args=(input_queue,output_queue))
    startx, starty = 0, 0

    cap = cv2.VideoCapture(camIdx)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    global points, screen_width, screen_height
    points = calibrate(cap)
    print(points)
    # points = [[157, 161], [152, 351], [484, 343], [480, 163]]
    getHomography(points)

    thread.start()
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            fps = cvFpsCalc.get()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            debug_image = copy.deepcopy(image)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            #push to other thread
            input_queue.put({"debug_image":debug_image,"multi_hand_landmarks":results.multi_hand_landmarks})
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    hand_sign_id = output_queue.get()
                    print("Output recieved")
                    if hand_sign_id == 2 or hand_sign_id == 3:  # Point gesture
                        cx, cy = hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y
                        cx, cy = cx * cap_width, cy*cap_height
                        cx, cy = normalizePoint(cx, cy)
                        cx, cy = dejitter(cx, cy)
                        cx = constrainVal(cx, screen_width)
                        cy = constrainVal(cy, screen_height)
                        wind_mouse(startx, starty, cx, cy,
                                       move_mouse=lambda x, y: mouse.move(x, y))
                        startx, starty = cx, cy
                        if hand_sign_id == 3:
                            print("press")
                            # mouse.press()
                        else:
                            print("release")
                            # mouse.release()
                    else:
                        pass

            image = normalizeImg(image)
            image = draw_fps(image, fps)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def on_mouse(event, x, y, flags, params):

    global points, count

    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 4:
            if count == 0:
                points.append([x, y])
            if count < 3:
                points.append([x, y])
            points[count] = [x, y]
            count += 1
    elif event == cv2.EVENT_MOUSEMOVE:
        if count == 0 or count == 4:
            pass
        else:
            points[count] = [x, y]


def calibrate(cap):

    global points, count

    waitTime = 50

    while (cap.isOpened()):

        _, frame = cap.read()

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', on_mouse)

        if count == 4:
            break

        for i in range(len(points)):
            if (i + 1 == len(points)):
                cv2.line(frame, points[i], points[0], (187, 87, 231), 2)
            else:
                cv2.line(frame, points[i], points[i+1], (187, 87, 231), 2)

        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(waitTime)

        if key == ord('b'):
            break
        elif key == ord('r'):
            count = 0
            points = []

    cv2.destroyWindow('Calibration')
    return points


def getHomography(points):
    global h, screen_width, screen_height
    pts_src = np.array(points)
    pts_dst = np.array([[0, 0], [0, screen_height], [
                       screen_width, screen_height], [screen_width, 0]])

    # Calculate Homography
    h, _ = cv2.findHomography(pts_src, pts_dst)


def normalizeImg(img):
    global h, screen_width, screen_height

    # Warp source image to destination based on homography
    return cv2.warpPerspective(
        img, h, (screen_width, screen_height))


def normalizePoint(x, y):
    global h
    pts = np.dot(h, np.array([x, y, 1.0]))
    pts = pts/pts[-1]
    return int(pts[0]), int(pts[1])


def dejitter(x, y):
    jitVarX = screen_width // 100
    jitVarY = screen_height // 100
    x = (x // jitVarX) * jitVarX
    y = (y // jitVarY) * jitVarY
    return x, y


def constrainVal(val, max, min=0):
    val = val if val >= min else min
    val = val if val < max else max
    return val


def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=15, D_0=12, move_mouse=lambda x, y: None):
    '''
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    '''
    sqrt3 = np.sqrt(3)
    sqrt5 = np.sqrt(5)

    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0
    dist = np.hypot(dest_x-start_x, dest_y-start_y)
    while dist >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + np.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))
        if current_x != move_x or current_y != move_y:
            # This should wait for the mouse polling interval
            current_x = move_x
            current_y = move_y
            move_mouse(current_x, current_y)
        dist = np.hypot(dest_x-start_x, dest_y-start_y)
    return current_x, current_y


def draw_fps(image, fps):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == "__main__":
    main()
