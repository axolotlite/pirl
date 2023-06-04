import cv2
import numpy as np
from cfg import CFG
# from screeninfo import get_monitors


class Homography(object):
    def __init__(self) -> None:
        self.points = []
        self.count = 0
        self.h_matrix = None
        self.pmon = CFG.monitors[CFG.handThreadScreen]
        self.s_width, self.s_height = self.pmon.width, self.pmon.height
        self.camIdx = CFG.camIdx

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.count < 4:
                if self.count == 0:
                    self.points.append([x, y])
                if self.count < 3:
                    self.points.append([x, y])
                self.points[self.count] = [x, y]
                self.count += 1
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.count == 0 or self.count == 4:
                pass
            else:
                self.points[self.count] = [x, y]

    def calibrate(self):
        waitTime = 50

        cap = cv2.VideoCapture(self.camIdx)
        if(CFG.MJPG):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # add this line
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.height)
        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while (cap.isOpened()):

            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            cv2.namedWindow('Calibration')
            cv2.moveWindow('Calibration', int(
                self.s_width * 1.3), int(self.s_height * 0.3))
            cv2.setMouseCallback('Calibration', self.on_mouse)

            if self.count == 4:
                break

            for i in range(len(self.points)):
                if (i + 1 == len(self.points)):
                    cv2.line(frame, self.points[i],
                             self.points[0], (187, 87, 231), 2)
                else:
                    cv2.line(
                        frame, self.points[i], self.points[i+1], (187, 87, 231), 2)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(waitTime)

            if key == ord('b'):
                break
            elif key == ord('r'):
                self.count = 0
                self.points = []
        cv2.destroyWindow('Calibration')
        cap.release()

    def get_homography(self):
        """Calculates the homography matrix and stores it in the class variable h_matrix
        """
        print(self.points)
        pts_src = np.array(self.points)
        pts_dst = np.array([[0, 0], [0, self.s_height], [
            self.s_width, self.s_height], [self.s_width, 0]])

        self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)

    def normalize_img(self, img):
        """Warps source image to destination based on homography

        Args:
            img (numpy.ndarray): Source image

        Returns:
            numpy.ndarray: Image warped based on h_matrix
        """
        return cv2.warpPerspective(
            img, self.h_matrix, (self.s_width, self.s_height))

    def normalize_point(self, x, y):
        """Finds the coordinates matching the source coordinates in the warped image

        Args:
            x (int): position on x-axis in original image
            y (int): position on y-axis in original image

        Returns:
            int, int: position on x-axis and y-axis in warped image
        """
        pts = np.dot(self.h_matrix, np.array([x, y, 1.0]))
        pts = pts/pts[-1]
        return int(pts[0]), int(pts[1])

    def dejitter(self, x, y):
        jitVarX = self.s_width // 100
        jitVarY = self.s_height // 100
        x = (x // jitVarX) * jitVarX
        y = (y // jitVarY) * jitVarY
        return x, y

    def constrain_val(self, val, max, min=0):
        val = val if val >= min else min
        val = val if val < max else max
        return val

    def process_point(self, x, y):
        x, y = self.normalize_point(x, y)
        x, y = self.dejitter(x, y)
        x = self.constrain_val(x, self.s_width)
        y = self.constrain_val(y, self.s_height)
        return x, y
