import cv2
import numpy as np
import screeninfo
from skimage.metrics import structural_similarity as ssim
import os


class Autocalibration:
    def __init__(self):
        self.black_screen = None
        self.white_screen = None
        self.points = []
        self.screen_id = 1
        self.screen = screeninfo.get_monitors()[self.screen_id]
        self.qt5_vars = []
        self.get_vars()
        self.delete_qt_vars()
        self.failure_condition = ord('q')
    # This is hacky code and needs to be made less dirty.

    def get_vars(self):
        for k, v in os.environ.items():
            if k.startswith("QT_") and "cv2" in v:
                self.qt5_vars.append([k, os.environ[k]])

    def delete_qt_vars(self):
        for k, _ in self.qt5_vars:
            del os.environ[k]

    def add_qt_vars(self):
        for k, var_data in self.qt5_vars:
            os.environ[k] = var_data

    def show_captures(self):
        self.add_qt_vars()
        cv2.imshow("Black Screen", self.black_screen)
        cv2.waitKey()
        cv2.imshow("White Screen", self.white_screen)
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.delete_qt_vars()

    def show_corners(self):
        image = self.white_screen.copy()
        for idx, point in enumerate(self.points):
            point = (int(point[0]), int(point[1]))
            image = cv2.putText(image, str(
                idx), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            image = cv2.circle(image, point, radius=0,
                               color=(0, 220, 0), thickness=10)
        cv2.imshow('Harris Corneres', image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key

    def on_failure(self, comparitor, func):
        if(comparitor == self.failure_condition):
            func()
            return True
        return False

    def capture_screen(self):
        """
        Captures a picture of a screen when it's white and another when it's black.

        Returns:
            black_screen,white_screen: Two pictures containing the computer screen once black and another white.
        """
        self.add_qt_vars()
        # Get the screen size

        screen_width, screen_height = self.screen.width, self.screen.height
        # create the background and leave it black
        background = np.zeros((screen_height, screen_width, 3), np.uint8)

        # Initialize the camera device
        cap = cv2.VideoCapture(0)

        # Create a max window on the selected screen
        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Window", background)
        cv2.moveWindow("Window", self.screen.x - 1, self.screen.y - 1)
        cv2.setWindowProperty(
            "Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(250)

        # Capture an image of the black screen
        _, self.black_screen = cap.read()

        # Change the color of the window to white
        background.fill(255)

        cv2.setWindowProperty(
            "Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Window", background)
        cv2.waitKey(250)

        # Capture another image this time of the white screen
        _, self.white_screen = cap.read()

        cv2.destroyWindow("Window")
        # Release the camera device
        cap.release()
        self.delete_qt_vars()

    def mask_screen_diff(self):
        """
        Creates a mask by finding the difference between the two images
        Args:
            black_screen, white_screen : Images of a white and black computer screens
        Returns:
            mask: a mask of the computer screen
        """
        # Convert the images to grayscale
        gray1 = cv2.cvtColor(self.black_screen, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.white_screen, cv2.COLOR_BGR2GRAY)
        # Compute the structural similarity index (SSIM) between the images
        (score, diff) = ssim(gray1, gray2, full=True)
        # Normalize the difference image to the range [0, 255]
        diff = (diff * 255).astype('uint8')

        # Apply a threshold to the difference image
        thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        # Apply a morphological operation to close small gaps in the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((10, 25), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Find contours in the morphological image
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the contours
        mask = np.zeros_like(gray1)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                cv2.drawContours(mask, [contour], 0, 255, -1)
        return(mask)

    def mask_screen_boundaries(self, image):
        """
        Creates a mask through thresholding and opening the image.
        Args:
            image: an image of a white computer screen
        Returns:
            mask: a mask of the computer screen
        """
        # Convert to grayscale
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.convertScaleAbs(mask)
        # Threshold the image to remove non-white pixels
        _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        # Perform morphological closing on the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Perform morphological opening on the binary image
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def Harris_Corner_Method(self, image, masks):
        """
        A wrapper for cv2 harris corner method that takes an image and an array of masks
        using them to return the corners of a screen
        Args:
            image: any image
            masks: an array of np.arrays that resembles masks.
        Returns:
            centers: array of 4 values denoting the corners of a screen
        """
        self.add_qt_vars()
        harris_corners = []
        for mask in masks:
            harris_corners.append(cv2.cornerHarris(mask, 9, 9, 0.05))
        # Select the array with the lowest mean
        harris_corners = min(harris_corners, key=lambda arr: arr.mean())
        # print(harris_corners)
        # creates a 3x3 square structuring element using the NumPy ones function.
        kernel = np.ones((3, 3), np.uint8)
        # dilates the corner points, using the kernel structuring element with 2 iterations
        harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)
        # Find all points fulfilling this condition
        locations = np.where(harris_corners > 0.09 * harris_corners.max())
        # invert then transpose it
        coords = np.vstack((locations[1], locations[0])).T
        _, centers = self.k_means(coords, 4, self.select_four_points(coords))
        self.delete_qt_vars()
        return(centers)

    def select_four_points(self, points):
        """
        Selects 4 points from an array of 2D points, such that the four points
        have the highest pairwise Euclidean distance between each other and are
        not close to each other.

        Args:
        - points (numpy array): An array of 2D points

        Returns:
        - selected_points (numpy array): An array of 4 selected points
        """
        # Compute pairwise Euclidean distances between all points
        distances = np.sqrt(
            ((points[:, np.newaxis] - points) ** 2).sum(axis=2))

        # Initialize selected points array and selected point index list
        selected_points = np.zeros((4, 2), dtype=int)
        selected_indices = []

        # Select the point with the highest maximum distance as the first selected point
        max_distances = distances.max(axis=1)
        selected_index = np.argmax(max_distances)
        selected_points[0] = points[selected_index]
        selected_indices.append(selected_index)

        # Select the point with the highest minimum distance from the first point
        min_distances = distances[selected_index]
        selected_index = np.argmax(min_distances)
        selected_points[1] = points[selected_index]
        selected_indices.append(selected_index)

        # Select the point with the highest minimum distance from the first two points
        distances_from_selected = distances[selected_indices].min(axis=0)
        selected_index = np.argmax(distances_from_selected)
        selected_points[2] = points[selected_index]
        selected_indices.append(selected_index)

        # Select the point with the highest minimum distance from the first three points
        distances_from_selected = distances[selected_indices].min(axis=0)
        # Add condition to check proximity
        mask = (distances_from_selected > 0.1 * np.mean(distances))
        filtered_indices = np.where(mask)[0]
        selected_index = filtered_indices[np.argmax(
            distances_from_selected[mask])]
        selected_points[3] = points[selected_index]

        return selected_points

    def k_means(self, X, k, centers=None, num_iter=100):
        """
        Implementation of Lloyd's algorithm for K-means clustering.
        Lloyd, S.(1982). Least squares quantization in PCM. IEEE Transactions On Information Theory 28 129–137.
        Args:
            X (numpy.ndarray, shape=(n_samples, n_dim)): The dataset
            k (int): Number of cluster
            num_iter (int): Number of iterations to run the algorithm
            centers (numpy.ndarray, shape=(k, )): Starting centers
        Returns:
            cluster_assignments (numpy.ndarray; shape=(n_samples, n_dim)): The clustering label for each data point
            centers (numpy.ndarray, shape=(k, )): The cluster centers
        """
        if centers is None:
            rnd_centers_idx = np.random.choice(
                np.arange(X.shape[0]), k, replace=False)
            centers = X[rnd_centers_idx]
        for _ in range(num_iter):
            distances = np.sum(
                np.sqrt((X - centers[:, np.newaxis]) ** 2), axis=-1)
            cluster_assignments = np.argmin(distances, axis=0)
            for i in range(k):
                msk = (cluster_assignments == i)
                centers[i] = np.mean(X[msk], axis=0) if np.any(
                    msk) else centers[i]

        return cluster_assignments, centers

    def order_points(self, points):
        corners = [
            [0, 0],
            [0, self.screen.height],
            [self.screen.width, self.screen.height],
            [self.screen.width, 0]
        ]
        ordered_points = []
        point_idx = 0
        pidxs = []
        for idx, corner in enumerate(corners):
            min_distance = self.screen.height * self.screen.width
            for pidx, point in enumerate(points):
                if pidx in pidxs:
                    continue
                euc_distance = ((point[0] - corner[0]) **
                                2 + (point[1] - corner[1])**2)**0.5
                if euc_distance < min_distance:
                    min_distance = euc_distance
                    current_point = point
                    point_idx = pidx
            ordered_points.append(current_point)
            pidxs.append(point_idx)
        print(ordered_points)
        print(pidxs)
        return ordered_points

    def autocalibrate(self):
        self.capture_screen()
        # black_screen = cv2.imread("test_1.jpg", cv2.IMREAD_COLOR)
        # white_screen = cv2.imread("test_2.jpg", cv2.IMREAD_COLOR)
        boundaries_mask = self.mask_screen_boundaries(self.white_screen)
        diff_mask = self.mask_screen_diff()
        points = self.Harris_Corner_Method(self.white_screen.copy(), [
                                           boundaries_mask, diff_mask])
        points = self.order_points(points)
        self.points = points


def main():
    test = Autocalibration()
    test.autocalibrate()
    test.show_corners()
    # test.show_captures()
    # cv2.imshow("mask",test.mask_screen_diff())


if __name__ == '__main__':
    main()
