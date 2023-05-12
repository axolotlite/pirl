import cv2
import numpy as np
from screeninfo import get_monitors


class Homography(object):
    def __init__(self) -> None:
        self.points = []
        self.count = 0
        self.h_matrix = None
        self.pmon = get_monitors()[0]
        self.s_width, self.s_height = self.pmon.width, self.pmon.height
        self.camIdx = 0
        
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
        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while (cap.isOpened()):

            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            cv2.namedWindow('Calibration')
            cv2.setMouseCallback('Calibration', self.on_mouse)

            if self.count == 4:
                break

            for i in range(len(self.points)):
                if (i + 1 == len(self.points)):
                    cv2.line(frame, self.points[i], self.points[0], (187, 87, 231), 2)
                else:
                    cv2.line(frame, self.points[i], self.points[i+1], (187, 87, 231), 2)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(waitTime)

            if key == ord('b'):
                break
            elif key == ord('r'):
                self.count = 0
                self.points = []
        cv2.destroyWindow('Calibration')
        cap.release()
        
        # pts_src = np.array(self.points)
        # pts_dst = np.array([[0, 0], [0, self.s_height], [
        #                 self.s_width, self.s_height], [self.s_width, 0]])

        # # Calculate Homography
        # self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
    def homography(self):
        print(self.points)
        pts_src = np.array(self.points)
        pts_dst = np.array([[0, 0], [0, self.s_height], [
                        self.s_width, self.s_height], [self.s_width, 0]])

        # Calculate Homography
        self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)

    def normalizeImg(self, img):
        # Warp source image to destination based on homography
        return cv2.warpPerspective(
            img, self.h_matrix, (self.s_width, self.s_height))


    def normalizePoint(self, x, y):
        pts = np.dot(self.h_matrix, np.array([x, y, 1.0]))
        pts = pts/pts[-1]
        return int(pts[0]), int(pts[1])


    def dejitter(self, x, y):
        jitVarX = self.s_width // 100
        jitVarY = self.s_height // 100
        x = (x // jitVarX) * jitVarX
        y = (y // jitVarY) * jitVarY
        return x, y


def main():
    pass

if __name__ == "__main__":
    main()