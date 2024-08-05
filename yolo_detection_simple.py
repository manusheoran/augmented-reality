import cv2
import numpy as np
from ultralytics import YOLO

def overlay_image_homography(frame, overlay_img, box):
    x1, y1, x2, y2 = box

    # Define the destination points based on the bounding box
    destination_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
    
    # Get the size of the overlay image
    h, w = overlay_img.shape[:2]
    source_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
    
    # Calculate the homography matrix
    homography_matrix, _ = cv2.findHomography(source_points, destination_points)
    
    # Warp the overlay image to the detected plane
    warped_overlay = cv2.warpPerspective(overlay_img, homography_matrix, (frame.shape[1], frame.shape[0]))
    
    # Create a mask of the overlay image
    mask = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Ensure the frame has 3 channels
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    # Ensure the warped overlay has 4 channels
    if warped_overlay.shape[2] == 3:
        warped_overlay = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2BGRA)
    
    # Black out the area of the overlay in the frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Take only the region of the overlay from the overlay image
    overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)
    
    # Put the overlay in the frame and modify the frame
    frame = cv2.add(frame_bg, overlay_fg)
    
    return frame

def run ():

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Load the overlay image
    overlay_img = cv2.imread('ar.jpeg', cv2.IMREAD_UNCHANGED)

    # Ensure the overlay image has an alpha channel
    if overlay_img.shape[2] != 4:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

    # Start video capture from the laptop camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        
        # Process the results
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            confidences = result.boxes.conf  # Get confidences
            class_ids = result.boxes.cls  # Get class IDs

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                xmin, ymin, xmax, ymax = map(int, box)
                
                # Check if the detected object is a cell phone (class_id should match your cell phone class)
                if model.names[int(class_id)] == 'cell phone':  # Replace with your cell phone class name
                    # Prepare overlay position
                    box = (xmin, ymin, xmax, ymax)
                    
                    # Overlay the image on the detected cell phone
                    frame = overlay_image_homography(frame, overlay_img, box)

        # Display the output
        cv2.imshow('Object Detection and Overlay', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
