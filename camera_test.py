import cv2
import numpy as np
import screeninfo
from skimage.metrics import structural_similarity as ssim
#code to get the screen
def caputre_screen():
    # Get the screen size
    screen_id = 0  # Change this value to select a different screen
    screen = screeninfo.get_monitors()[screen_id]
    screen_width, screen_height = screen.width, screen.height
    white_background = np.zeros((screen_height, screen_width, 3), np.uint8)
    white_background.fill(255)

    # Initialize the camera device
    cap = cv2.VideoCapture(0)

    # Create a window on the selected screen
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Window", np.zeros((screen_height, screen_width, 3), np.uint8))
    cv2.moveWindow("Window", screen.x - 1, screen.y - 1)
    cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(250)

    # Capture an image
    _, black_screen = cap.read()

    # Change the color of the window
    cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Window", white_background)
    cv2.waitKey(250)

    # Capture another image
    _, white_screen = cap.read()

    cv2.destroyWindow("Window")
    # Release the camera device
    cap.release()
    return(black_screen,white_screen)
def mask_screen(img1, img2):

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray1", gray1)
    # cv2.imshow("gray2", gray2)
    # Compute the structural similarity index (SSIM) between the images
    (score, diff) = ssim(gray1, gray2, full=True)
    # Normalize the difference image to the range [0, 255]
    diff = (diff * 255).astype('uint8')

    # Apply a threshold to the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Apply a morphological operation to close small gaps in the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((10,25), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Find contours in the morphological image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the contours
    mask = np.zeros_like(gray1)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(mask, [contour], 0, 255, -1)
    return(mask)
def mask_screen2(image):
    # Convert to grayscale and convert to 8-bit unsigned format
    gray_screen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_screen = cv2.convertScaleAbs(gray_screen)
    cv2.imshow('gray_screen', gray_screen)
    cv2.waitKey(0)
    cv2.destroyWindow("gray_screen")
    # Perform morphological opening on the binary image
    _, img1 = cv2.threshold(gray_screen, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow('img1', img1)
    cv2.waitKey(0)
    cv2.destroyWindow("img1")
    kernel = np.ones((15,15), np.uint8)
    opened_image = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    return opened_image
def mask_selector():
    return
# Code to find line intersections. From https://stackoverflow.com/a/20677983
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def method1(im1, im2):
    #Get the mask
    mask = mask_screen(im1,im2)
    _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    # Filtering to improve the thresholded image
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Calculate contours and find the largest one
    cnts, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max([c for c in cnts], key=lambda x: cv2.contourArea(x))

    cv2.drawContours(im1, [cnt], 0, (0, 255, 0), 3)

    # Remove the concavities
    hull = cv2.convexHull(cnt)
    cv2.drawContours(im1, [hull], 0, (255, 0, 0), 2)
    hull = [tuple(p[0]) for p in hull]

    # Find all the corners
    tr = max(hull, key=lambda x: x[0] - x[1])
    cv2.circle(im1, tr, 3, (0, 0, 255), -1)

    tl = min(hull, key=lambda x: x[0] + x[1])
    cv2.circle(im1, tl, 3, (0, 0, 255), -1)

    br = max(hull, key=lambda x: x[0] + x[1])
    cv2.circle(im1, br, 3, (0, 0, 255), -1)

    bl = min(hull, key=lambda x: x[0] - x[1])
    cv2.circle(im1, bl, 3, (0, 0, 255), -1)

    cv2.imshow('im1', im1)
    cv2.imshow('thresh', thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def method2(im1,im2):
    mask = mask_screen(im1,im2)

    # Canny and Hough lines
    c = cv2.Canny(mask, 89, 200)
    lines = cv2.HoughLines(c, 1, np.pi / 180, 100, None, 0, 0)

    pts = []

    # Create segments for each line
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = np.array([int(x0 + 1000 * (-b)), int(y0 + 1000 * a)])
            pt2 = np.array([int(x0 - 1000 * (-b)), int(y0 - 1000 * a)])

            if not any([np.linalg.norm(pt1 - p[0]) < 100 for p in pts]):    # Filter out lines too close to each other
                pts.append(np.array([pt1, pt2]))

                cv2.line(im1, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA)

    for pt in pts:
        for comp in pts:
            intersect = np.array(line_intersection(pt, comp))
            if any(intersect < 0) or intersect[0] > im1.shape[1] or intersect[1] > im1.shape[0]:    # Filter out off-screen intersections
                continue

            intersect = np.asarray(intersect, dtype=int)
            print(intersect)
            cv2.circle(im1, tuple(intersect), 3, (0, 255, 0), -1)

    cv2.imshow('im1', im1)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Harris_Corner_Method(image, mask):
    """
    applies the Harris corner detection algorithm to the input mask image with:
    a blockSize of 9
    a ksize of 9
    a k value of 0.05
    The output is a grayscale image with each pixel value indicating the corner strength.
    """
    harris_corners = cv2.cornerHarris(mask, 9, 9, 0.05)
    # print(harris_corners)
    #creates a 3x3 square structuring element using the NumPy ones function.
    kernel = np.ones((3,3), np.uint8)
    #dilates the corner points, using the kernel structuring element with 2 iterations.
    #This is done to enhance the corner points and make them more visible.
    #TODO test different iterations
    harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)
    """
    marks the corner points on the original image with a specific color.
    It selects the pixels in image where the corresponding pixel is greater than 0.05 
    times the maximum pixel value in harris_corners
    then changes their colors in the image. This highlights the detected corner points in the image.
    """
    #Find all points fulfilling this condition
    locations = np.where(harris_corners > 0.09 * harris_corners.max())
    #invert then transpose it
    coords = np.vstack((locations[1],locations[0])).T
    _, centers = k_means(coords, 4)
    #Draw the centers
    for point in centers:
        image = cv2.circle(image, (int(point[0]),int(point[1])), radius=0, color=(0, 220, 0), thickness=10)
    cv2.imshow('Harris Corneres', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import numpy as np

def select_four_points(points):
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
    distances = np.sqrt(((points[:, np.newaxis] - points)**2).sum(axis=2))

    # Initialize selected points array and selected point index list
    selected_points = np.zeros((4, 2))
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
    mask = (distances_from_selected > 0.1*np.mean(distances))  # Add condition to check proximity
    filtered_indices = np.where(mask)[0]
    selected_index = filtered_indices[np.argmax(distances_from_selected[mask])]
    selected_points[3] = points[selected_index]

    return selected_points.astype(int)

def k_means(X, k, centers=None, num_iter=100):
    if centers is None:
        centers = select_four_points(X)
    for _ in range(num_iter):
        distances = np.sum(np.sqrt((X - centers[:, np.newaxis]) ** 2), axis=-1)
        cluster_assignments = np.argmin(distances, axis=0)
        for i in range(k):
            msk = (cluster_assignments == i)
            centers[i] = np.mean(X[msk], axis=0) if np.any(msk) else centers[i]

    return cluster_assignments, centers
def main():
    black_screen,white_screen = caputre_screen()
    # cv2.imshow('white_screen', white_screen)
    # cv2.waitKey(0)
    # cv2.destroyWindow("white_screen")
    # black_screen = cv2.imread("test_1.jpg", cv2.IMREAD_COLOR)
    # white_screen = cv2.imread("test_2.jpg", cv2.IMREAD_COLOR)
    mask = mask_screen2(white_screen)
    # mask = mask_screen(black_screen,white_screen)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyWindow("mask")
    Harris_Corner_Method(white_screen,mask)
    # Detect corners using the Harris corner detector
    # harris_corners = cv2.cornerHarris(opened_image, 9, 9, 0.05)

    # opened_image = 255 - opened_image
    # Harris_Corner_Method(white_screen,opened_image)
    # imgray = cv2.cvtColor(black_screen, cv2.COLOR_BGR2GRAY)

    # _, img2 = cv2.threshold(imgray, 127, 255,0)

    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # cv2.destroyWindow("img2")

    # mask = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyWindow("mask")

    # cv2.imshow('black_screen', black_screen)
    # cv2.waitKey(0)
    # cv2.destroyWindow("black_screen")
    # cv2.imshow('white_screen', white_screen)
    # cv2.waitKey(0)
    # cv2.destroyWindow("white_screen")
    # mask = mask_screen(black_screen,white_screen)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyWindow("mask")

    # method1(im1,im2)
    # method2(im1,im2)
    # Harris_Corner_Method(im2,mask)
    # Harris_Corner_Method(im1,mask)

if __name__ == '__main__':
    main()
