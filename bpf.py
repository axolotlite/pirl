from collections import namedtuple
from math import gcd
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# All values are in pixel. The region is a square of size 'size' pixels
CropRegion = namedtuple(
    'CropRegion', ['xmin', 'ymin', 'xmax',  'ymax', 'size'])

# Dictionary that maps from joint names to keypoint indices.
BODY_KP = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}


def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a-b)


class Body:
    def __init__(self, keypoints_norm=None, scores=None, score_thresh=None, crop_region=None, next_crop_region=None):
        """
        Attributes:
        scores : scores of the keypoints
        keypoints_norm : keypoints normalized ([0,1]) coordinates (x,y) in the squared cropped region
        keypoints_square : keypoints coordinates (x,y) in pixels in the square padded image
        keypoints : keypoints coordinates (x,y) in pixels in the source image (not padded)
        score_thresh : score threshold used
        crop_region : cropped region on which the current body was inferred
        next_crop_region : cropping region calculated from the current body keypoints and that will be used on next frame
        """
        self.keypoints_norm = keypoints_norm
        self.scores = scores
        self.score_thresh = score_thresh
        self.crop_region = crop_region
        self.next_crop_region = next_crop_region
        # self.keypoints = (np.array([self.crop_region.xmin, self.crop_region.ymin]) +
        #                 self.keypoints_norm * self.crop_region.size).astype(np.int32)
        self.keypoints = np.array([(kp[0] * 1280, kp[1] * 720) for kp in keypoints_norm])

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

    def distance_to_wrist(self, hand, wrist_handedness, pad_w=0, pad_h=0):
        """
        Calculate the distance between a hand (class Hand) wrist position 
        and one of the body wrist given by wrist_handedness (= "left" or "right")
        As the hand.landmarks cooordinates are expressed in the padded image, we must substract the padding (given by pad_w and pad_w)
        to be coherent with the body keypoint coordinates which are expressed in the source image.
        """
        return distance(hand.landmarks[0]-np.array([pad_w, pad_h]), self.keypoints[BODY_KP[wrist_handedness+'_wrist']])


class BodyPreFocusing:
    """
    Body Pre Focusing with Movenet
    Contains all is needed for :
    - Movenet smart cropping (determines from the body detected in frame N, 
    the region of frame N+1 ow which the Movenet inference is run). 
    - Body Pre Focusing (determining from the Movenet wrist keypoints a smaller zone
    on which Palm detection is run).
    Both Smart cropping and Body Pre Focusing are important for model accuracy when 
    the body is far.
    """

    def __init__(self, img_w, img_h, pad_w, pad_h, frame_size, mode="group", score_thresh=0.2, scale=1.0, hands_up_only=True):

        self.img_w = img_w
        self.img_h = img_h
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.frame_size = frame_size
        self.mode = mode
        self.score_thresh = score_thresh
        self.scale = scale
        self.hands_up_only = hands_up_only
        # Defines the default crop region (pads the full image from both sides to make it a square image)
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        self.init_crop_region = CropRegion(-self.pad_w, -self.pad_h, -self.pad_w +
                                           self.frame_size, -self.pad_h+self.frame_size, self.frame_size)

    """
    Smart cropping stuff
    """

    def crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[int(max(0, crop_region.ymin)):int(min(self.img_h, crop_region.ymax)), int(
            max(0, crop_region.xmin)):int(min(self.img_w, crop_region.xmax))]

        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary
            cropped = cv2.copyMakeBorder(cropped,
                                         int(max(0, -crop_region.ymin)),
                                         int(max(0, crop_region.ymax-self.img_h)),
                                         int(max(0, -crop_region.xmin)),
                                         int(max(0, crop_region.xmax-self.img_w)),
                                         cv2.BORDER_CONSTANT)

        cropped = cv2.resize(
            cropped, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        return cropped

    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.
        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[BODY_KP['left_hip']] > self.score_thresh or
                scores[BODY_KP['right_hip']] > self.score_thresh) and
                (scores[BODY_KP['left_shoulder']] > self.score_thresh or
                scores[BODY_KP['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, body, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.
        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['left_shoulder',
                        'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - body.keypoints[BODY_KP[joint]][1])
            dist_x = abs(center_x - body.keypoints[BODY_KP[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(BODY_KP)):
            if body.scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - body.keypoints[i][1])
            dist_x = abs(center_x - body.keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.
        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[BODY_KP['left_hip']]
                        [0] + body.keypoints[BODY_KP['right_hip']][0]) // 2
            center_y = (body.keypoints[BODY_KP['left_hip']]
                        [1] + body.keypoints[BODY_KP['right_hip']][1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(
                body, center_x, center_y)
            crop_length_half = np.amax(
                [max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x,
                           center_y, self.img_h - center_y])
            crop_length_half = int(
                round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half,
                           center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length, crop_length)
        else:
            return self.init_crop_region

    """
    Body Pre Focusing stuff
    """

    def estimate_focus_zone_size(self, body):
        """
        This function is called if at least the segment "wrist_elbow" is visible.
        We calculate the length of every segment from a predefined list. A segment length
        is the distance between the 2 endpoints weighted by a coefficient. The weight have been chosen
        so that the length of all segments are roughly equal. We take the maximal length to estimate
        the size of the focus zone. 
        If no segment are vissible, we consider the body is very close 
        to the camera, and therefore there is no need to focus. Return 0
        To not have at least one shoulder and one hip visible means the body is also very close
        and the estimated size needs to be adjusted (bigger)
        """
        segments = [
            ("left_shoulder", "left_elbow", 2.3),
            ("left_elbow", "left_wrist", 2.3),
            ("left_shoulder", "left_hip", 1),
            ("left_shoulder", "right_shoulder", 1.5),
            ("right_shoulder", "right_elbow", 2.3),
            ("right_elbow", "right_wrist", 2.3),
            ("right_shoulder", "right_hip", 1),
        ]
        lengths = []
        for s in segments:
            if body.scores[BODY_KP[s[0]]] > self.score_thresh and body.scores[BODY_KP[s[1]]] > self.score_thresh:
                l = np.linalg.norm(
                    body.keypoints[BODY_KP[s[0]]] - body.keypoints[BODY_KP[s[1]]])
                lengths.append(l)
        if lengths:
            if (body.scores[BODY_KP["left_hip"]] < self.score_thresh and
                body.scores[BODY_KP["right_hip"]] < self.score_thresh or
                body.scores[BODY_KP["left_shoulder"]] < self.score_thresh and
                    body.scores[BODY_KP["right_shoulder"]] < self.score_thresh):
                coef = 1.5
            else:
                coef = 1.0
            # The size is made even
            return 2 * int(coef * self.scale * max(lengths) / 2)
        else:
            return 0

    def get_focus_zone(self, body):
        """
        Return a tuple (focus_zone, label)
        'body' = instance of class Body
        'focus_zone' is a zone around a hand or hands, depending on the value 
        of self.mode ("left", "right", "higher" or "group") and on the value of self.hands_up_only.
            - self.mode = "left" (resp "right"): we are looking for the zone around the left (resp right) wrist,
            - self.mode = "group": the zone encompasses both wrists,
            - self.mode = "higher": the zone is around the higher wrist (smaller y value),
            - self.hands_up_only = True: we don't take into consideration the wrist if the corresponding elbow is above the wrist,
        focus_zone is a list [left, top, right, bottom] defining the top-left and right-bottom corners of a square. 
        Values are expressed in pixels in the source image C.S.
        The zone is constrained to the squared source image (= source image with padding if necessary). 
        It means that values can be negative.
        left and right in [-pad_w, img_w + pad_w]
        top and bottom in [-pad_h, img_h + pad_h]
        'label' describes which wrist keypoint(s) were used to build the zone : "left", "right" or "group" (if built from both wrists)

        If the wrist keypoint(s) is(are) not present or is(are) present but self.hands_up_only = True and
        wrist(s) is(are) below corresponding elbow(s), then focus_zone = None.
        """

        def zone_from_center_size(x, y, size):
            """
            Return zone [left, top, right, bottom] 
            from zone center (x,y) and zone size (the zone is square).
            """
            half_size = size // 2
            size = half_size * 2
            if size > self.img_w:
                x = self.img_w // 2
            x1 = x - half_size
            if x1 < -self.pad_w:
                x1 = -self.pad_w
            elif x1 + size > self.img_w + self.pad_w:
                x1 = self.img_w + self.pad_w - size
            x2 = x1 + size
            if size > self.img_h:
                y = self.img_h // 2
            y1 = y - half_size
            if y1 < -self.pad_h:
                y1 = -self.pad_h
            elif y1 + size > self.img_h + self.pad_h:
                y1 = self.img_h + self.pad_h - size
            y2 = y1 + size
            return [x1, y1, x2, y2]

        def get_one_hand_zone(hand_label):
            """
            Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
            Values are expressed in pixels in the source image C.S.
            If the wrist keypoint is not visible, return None.
            If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
            """
            wrist_kp = hand_label + "_wrist"
            wrist_score = body.scores[BODY_KP[wrist_kp]]
            if wrist_score < self.score_thresh:
                return None
            x, y = body.keypoints[BODY_KP[wrist_kp]]
            if self.hands_up_only:
                # We want to detect only hands where the wrist is above the elbow (when visible)
                elbow_kp = hand_label + "_elbow"
                if body.scores[BODY_KP[elbow_kp]] > self.score_thresh and \
                        body.keypoints[BODY_KP[elbow_kp]][1] < body.keypoints[BODY_KP[wrist_kp]][1]:
                    return None
            # Let's evaluate the size of the focus zone
            size = self.estimate_focus_zone_size(body)
            if size == 0:
                # The hand is too close. No need to focus
                return [-self.pad_w, -self.pad_h, self.frame_size-self.pad_w, self.frame_size-self.pad_h]
            return zone_from_center_size(x, y, size)

        if self.mode == "group":
            zonel = get_one_hand_zone("left")
            if zonel:
                zoner = get_one_hand_zone("right")
                if zoner:
                    xl1, yl1, xl2, yl2 = zonel
                    xr1, yr1, xr2, yr2 = zoner
                    x1 = min(xl1, xr1)
                    y1 = min(yl1, yr1)
                    x2 = max(xl2, xr2)
                    y2 = max(yl2, yr2)
                    # Global zone center (x,y)
                    x = int((x1+x2)/2)
                    y = int((y1+y2)/2)
                    size_x = x2-x1
                    size_y = y2-y1
                    size = 2 * (max(size_x, size_y) // 2)
                    return (zone_from_center_size(x, y, size), "group")
                else:
                    return (zonel, "left")
            else:
                return (get_one_hand_zone("right"), "right")
        elif self.mode == "higher":
            if body.scores[BODY_KP["left_wrist"]] > self.score_thresh:
                if body.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    if body.keypoints[BODY_KP["left_wrist"]][1] > body.keypoints[BODY_KP["right_wrist"]][1]:
                        hand_label = "right"
                    else:
                        hand_label = "left"
                else:
                    hand_label = "left"
            else:
                if body.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    hand_label = "right"
                else:
                    return (None, None)
            return (get_one_hand_zone(hand_label), hand_label)
        else:  # "left" or "right"
            return (get_one_hand_zone(self.mode), self.mode)


def main():
    # For webcam input:
    cap = cv2.VideoCapture("video.mp4")

    bpf = BodyPreFocusing(img_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), img_h=int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), pad_w=0, pad_h=0, frame_size=640)
    
    crop_region = bpf.init_crop_region

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (pd_input_length, pd_input_length))

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the pose annotation on the image.
            annotated = image.copy()
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if results.pose_landmarks:
                scores = [
                    landmark.visibility for landmark in results.pose_landmarks.landmark]
                keypoints = np.array([(landmark.x, landmark.y)
                                           for landmark in results.pose_landmarks.landmark])

            body = Body(keypoints, scores, score_thresh=0.2,
                crop_region=crop_region)

            crop_region = bpf.determine_crop_region(body)
            cropped = bpf.crop_and_resize(
                image, crop_region)

            # out.write(cropped)

            cv2.imshow('MediaPipe Pose', annotated)
            cv2.imshow('Cropped', cropped)
        #   cv2.waitKey(0)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    # out.release()


if __name__ == '__main__':
    main()
