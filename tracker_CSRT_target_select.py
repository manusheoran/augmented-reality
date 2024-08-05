import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the ROI variables
roi_defined = False
roi = None
original_roi = None  # To store the original ROI coordinates
saved_roi = None  # To store the saved ROI image
frame = None  # Declare frame as global

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()

# Load the local image for overlay
local_image_path = 'ar.jpeg'  # Change this to your local image path
overlay_image = cv2.imread(local_image_path, cv2.IMREAD_COLOR)

if overlay_image is None:
    print(f"Error loading image from {local_image_path}")
    cap.release()
    cv2.destroyAllWindows()
    exit()

def overlay_image_on_frame(frame, roi, original_roi):
    """Overlay a local image on the specified ROI."""
    if roi and original_roi and overlay_image is not None:
        x, y, w, h = roi
        original_x, original_y, original_w, original_h = original_roi
        
        # Resize the overlay image based on the original ROI size
        overlay_resized = cv2.resize(overlay_image, (original_w, original_h))
        
        # Ensure the overlay image fits within the frame
        if (x + original_w <= frame.shape[1]) and (y + original_h <= frame.shape[0]):
            # Extract the region of interest from the frame
            frame_roi = frame[y:y + original_h, x:x + original_w]
            
            # Blend the overlay with the frame ROI
            alpha = 1.0  # Full opacity
            cv2.addWeighted(overlay_resized, alpha, frame_roi, 1 - alpha, 0, frame_roi)
            
            # Place the updated ROI back into the frame
            frame[y:y + original_h, x:x + original_w] = frame_roi

def draw_roi(event, x, y, flags, param):
    global roi_defined, x1, y1, roi, original_roi, saved_roi, tracker, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the starting point of the rectangle
        roi_defined = True
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize the rectangle
        roi_defined = False
        x2, y2 = x, y
        roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        
        # Save the ROI image and its original position
        if roi[2] > 0 and roi[3] > 0 and frame is not None:
            saved_roi = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy()
            original_roi = roi  # Store the original ROI coordinates
            cv2.setMouseCallback('Object Detection', lambda *args : None)
            # Initialize the tracker with the ROI
            tracker.init(frame, (roi[0], roi[1], roi[2], roi[3]))
            

def run():
    global roi, frame  # Declare roi and frame as global to modify their values

    # Set up the mouse callback
    cv2.namedWindow('Object Detection')
    cv2.setMouseCallback('Object Detection', draw_roi)

    while True:
        # Capture a frame from the video
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Update the tracker if ROI is defined
        if original_roi is not None:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                roi = (x, y, w, h)
                
                # Overlay the image on the tracked ROI using the original size
                overlay_image_on_frame(frame, roi, original_roi)
            else:
                print("Tracking failed")

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
