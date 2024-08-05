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

def create_mesh_overlay(frame, box, color=(0, 255, 0), thickness=1, mesh_size=10):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Ensure coordinates are integers

    # Create an empty image with the same dimensions as the frame
    mesh = np.zeros_like(frame)

    # Draw horizontal lines
    for y in range(y1, y2, mesh_size):
        cv2.line(mesh, (x1, y), (x2, y), color, thickness)

    # Draw vertical lines
    for x in range(x1, x2, mesh_size):
        cv2.line(mesh, (x, y1), (x, y2), color, thickness)

    # Create a mask of the mesh
    mesh_gray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)
    _, mesh_mask = cv2.threshold(mesh_gray, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mesh_mask_inv = cv2.bitwise_not(mesh_mask)

    # Black out the area of the mesh in the frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=mesh_mask_inv)

    # Take only the region of the mesh from the mesh image
    mesh_fg = cv2.bitwise_and(mesh, mesh, mask=mesh_mask)

    # Add the mesh to the frame
    frame = cv2.add(frame_bg, mesh_fg)

    return frame

def stackImages(imgArray, scale, labels=[]):
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def run():
    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)

    print("Press Enter to capture the target image...")

    # Flag to indicate when to capture the image
    capture_image = False
    imgTarget = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow('Capture Target Image', frame)

        # Capture the image when Enter key is pressed
        if cv2.waitKey(1) & 0xFF == 13:  # Enter key
            imgTarget = frame
            capture_image = True
            break

    print("Target image captured. Press Enter again to start video feed...")
    cv2.waitKey(0)  # Wait for any key press

    # Apply preprocessing on the captured image
    imgTargetProcessed = preprocess_image_method3(imgTarget)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(imgTargetProcessed, None)

    # Create Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    detection = False

    while True:
        success, imgWebcam = cap.read()
        if not success:
            break

        imgWebcamProcessed = preprocess_image_method3(imgWebcam)
        imgAug = imgWebcam.copy()
        kp2, des2 = orb.detectAndCompute(imgWebcamProcessed, None)

        # Default value for imgStacked
        imgStacked = imgWebcam.copy()

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        imgFeatures = cv2.drawMatches(imgTargetProcessed, kp1, imgWebcamProcessed, kp2, good, None, flags=2)
        
        if len(good) > 20:
            detection = True
            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            pts = np.float32([[0, 0], [0, imgTarget.shape[0]], [imgTarget.shape[1], imgTarget.shape[0]], [imgTarget.shape[1], 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            box = (dst[0][0][0], dst[0][0][1], dst[2][0][0], dst[2][0][1])

            # Create and apply the mesh overlay
            imgAug = create_mesh_overlay(imgWebcam, box)

            imgStacked = stackImages(([imgWebcam, imgTarget], [imgFeatures, imgAug]), 0.5)

        cv2.imshow('imgStacked', imgStacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
