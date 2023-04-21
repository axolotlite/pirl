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
    white_screen = np.zeros((screen_height, screen_width, 3), np.uint8)
    white_screen.fill(255)

    # Initialize the camera device
    cap = cv2.VideoCapture(0)

    # Create a window on the selected screen
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Window", np.zeros((screen_height, screen_width, 3), np.uint8))
    cv2.moveWindow("Window", screen.x - 1, screen.y - 1)
    cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(250)

    # Capture an image
    _, img1 = cap.read()

    # Change the color of the window
    cv2.setWindowProperty("Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Window", white_screen)
    cv2.waitKey(250)

    # Capture another image
    _, img2 = cap.read()

    cv2.destroyWindow("Window")
    # Release the camera device
    cap.release()
    return(img1,img2)
def mask_screen(img1, img2):

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the structural similarity index (SSIM) between the images
    (score, diff) = ssim(gray1, gray2, full=True)
    # Normalize the difference image to the range [0, 255]
    diff = (diff * 255).astype('uint8')

    # Apply a threshold to the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Apply a morphological operation to close small gaps in the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the morphological image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the contours
    mask = np.zeros_like(gray1)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(mask, [contour], 0, 255, -1)
    return(mask)
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
    harris_corners = cv2.cornerHarris(mask, 9, 9, 0.05)
    print(harris_corners)

    kernel = np.ones((3,3), np.uint8)
    harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)

    image[harris_corners > 0.05 * harris_corners.max()] = [255, 127, 127]
    cv2.imshow('Harris Corneres', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    im1,im2 = caputre_screen()
    mask = mask_screen(im1,im2)
    # method1(im1,im2)
    # method2(im1,im2)
    Harris_Corner_Method(im2,mask)
    Harris_Corner_Method(im1,mask)

if __name__ == '__main__':
    main()
