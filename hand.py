import cv2
import mediapipe as mp
import numpy as np
import threading
import mouse
# import voice
#mediapipe variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
#detection specific variables
is_holistic = True
model_complexity=1
min_detection_confidence=0.5
min_tracking_confidence=0.5

points = []
count = 0
threshold = 0
thresFlag = False
right = False
h = None
camIdx = 0
sideIdx = 1
screen_width = 1920
screen_height = 1080
smoothFactor = 13


def main():
    cap = cv2.VideoCapture(camIdx)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    global points, screen_width, screen_height
    points = calibrate(cap)
    if(is_holistic):
        with mp_holistic.Holistic(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    landmark = results.right_hand_landmarks.landmark[8]
                    cx, cy = landmark.x * cap_width, landmark.y * cap_height
                    cx, cy = normalizePoint(cx, cy)
                    if 0 <= cx < screen_width and 0 <= cy < screen_height:
                        # mouse.move(cx,cy,duration = 0.05)
                        cv2.circle(image, (cx, cy), 5, (187, 87, 231), 2)
                
                cv2.imshow('MediaPipe Hands', normalizeImg(image))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    else:
        with mp_hands.Hands(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

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
                        landmark = hand_landmarks.landmark[8]
                        cx, cy = landmark.x * cap_width, landmark.y * cap_height
                        cx, cy = normalizePoint(cx, cy)
                        if 0 <= cx < screen_width and 0 <= cy < screen_height:
                            # mouse.move(cx,cy,duration = 0.05)
                            cv2.circle(image, (cx, cy), 5, (187, 87, 231), 2)
                # Flip the image horizontally for a selfie-view display.
                # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                cv2.imshow('MediaPipe Hands', normalizeImg(image))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


def side():
    cap = cv2.VideoCapture(sideIdx)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    global threshold
    calibrateSide(cap)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

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
                    landmark = hand_landmarks.landmark[8]
                    # Tip of pointer finger only
                    if (right and landmark.x*cap_width >= threshold) or (not right and landmark.x*cap_width <= threshold):
                        mouse.press()
                    else:
                        mouse.release()
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            cv2.imshow('Side View', image)
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

        grabbed, frame = cap.read()

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

    getHomography(points)
    cv2.destroyWindow('Calibration')
    return points


def sideMouse(event, x, y, flags, params):
    global threshold, thresFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        thresFlag = True
    elif event == cv2.EVENT_MOUSEMOVE:
        threshold = x


def calibrateSide(cap):
    waitTime = 50

    while (cap.isOpened()):

        grabbed, frame = cap.read()

        cv2.namedWindow('Side Calibration')
        cv2.setMouseCallback('Side Calibration', sideMouse)

        if thresFlag:
            break
        
        if threshold != 0:
            cv2.line(frame, [threshold, 0], [threshold, screen_height], (187, 87, 231), 2)

        cv2.imshow('Side Calibration', frame)
        key = cv2.waitKey(waitTime)

        if key == ord('b'):
            break

    cv2.destroyWindow('Side Calibration')


def getHomography(points):
    global h, screen_width, screen_height
    pts_src = np.array(points)
    pts_dst = np.array([[0, 0], [0, screen_height], [
                       screen_width, screen_height], [screen_width, 0]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)


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


if __name__ == "__main__":
    # voice = threading.Thread(target=voice.main)
    # voice.start()
    main = threading.Thread(target=main)
    side = threading.Thread(target=side)
    main.start()
    side.start()
    main.join()
    side.join()
