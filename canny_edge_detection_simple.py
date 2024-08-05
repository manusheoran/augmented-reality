import cv2
import numpy as np

def preprocess_image_method3(img):
    """
    Uses a different kernel size for dilation and erosion to preprocess the image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
    img_canny = cv2.Canny(img_blur, 30, 100)
    
    kernel = np.ones((7, 7))
    img_dilated = cv2.dilate(img_canny, kernel, iterations=3)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
    
    return img_eroded


def draw_mesh_on_rectangle(image, rect_points, mesh_size=20):
    """
    Draws a mesh grid over a rectangular area defined by rect_points.
    """
    if len(rect_points) != 4:
        return image

    # Create a mask for the rectangle
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [rect_points], -1, (255, 255, 255), -1)

    # Draw mesh on the mask
    x, y, w, h = cv2.boundingRect(rect_points)
    for i in range(x, x + w, mesh_size):
        cv2.line(mask, (i, y), (i, y + h), (0, 255, 0), 1)
    for j in range(y, y + h, mesh_size):
        cv2.line(mask, (x, j), (x + w, j), (0, 255, 0), 1)

    # Combine the mesh with the original image
    image_with_mesh = cv2.addWeighted(image, 1, mask, 0.5, 0)
    return image_with_mesh

def run():
    # Open the default camera (usually the first camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        
        eroded_edges = preprocess_image_method3(frame)

        # Find contours from the edges
        contours, _ = cv2.findContours(eroded_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Flag to check if any rectangle is detected
        rectangle_detected = False

        # Iterate through the contours
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon is a rectangle and large enough
            if len(approx) == 4 and cv2.contourArea(contour) > 500:
                rectangle_detected = True
                # Draw the rectangle on the original frame
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                # Draw mesh over the detected rectangle
                frame = draw_mesh_on_rectangle(frame, approx)

        # Display the frame with mesh or normal frame
        cv2.imshow('Detected Rectangles with Mesh', frame if rectangle_detected else frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

