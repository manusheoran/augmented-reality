import cv2
import numpy as np

# Global variables to store the rectangle coordinates and ROI
rect_start = None
rect_end = None
drawing = False
roi = None

# Load the local image for overlay
local_image_path = 'ar.jpeg'  # Change this to your local image path
overlay_image = cv2.imread(local_image_path, cv2.IMREAD_COLOR)

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        
        drawing = False
        rect_end = (x, y)
        cv2.setMouseCallback('Object Detection', lambda *args : None)

def extract_roi(image, rect_start, rect_end):
    x1, y1 = rect_start
    x2, y2 = rect_end
    
    # Ensure the rectangle coordinates are in the correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Extract the region of interest (ROI)
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return None, None, None, None
    
    global roi
    roi = image[y1:y2, x1:x2]
    return x1, y1, x2, y2

def draw_mesh(image, rect_start, rect_end):
    x1, y1 = rect_start
    x2, y2 = rect_end
    
    # Ensure the rectangle coordinates are in the correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Draw a grid mesh overlay on the rectangle
    grid_spacing = 20
    for x in range(x1, x2, grid_spacing):
        cv2.line(image, (x, y1), (x, y2), (255, 0, 0), 1)  # Vertical lines
    for y in range(y1, y2, grid_spacing):
        cv2.line(image, (x1, y), (x2, y), (255, 0, 0), 1)  # Horizontal lines

    # Draw the rectangle
    cv2.rectangle(image, rect_start, rect_end, (0, 255, 0), 2)
    
    return image
    
def overlay_image_roi(image, rect_start, rect_end, alpha=1.0):
   
    if overlay_image is not None:
        x1, y1 = rect_start
        x2, y2 = rect_end
        
        # Ensure the rectangle coordinates are in the correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Define the ROI based on the rectangle coordinates
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # Resize the overlay image to fit the ROI size
        overlay_resized = cv2.resize(overlay_image, (w, h))
        
        # Ensure the overlay image fits within the frame
        if (x + w <= image.shape[1]) and (y + h <= image.shape[0]):
            # Extract the region of interest from the frame
            frame_roi = image[y:y + h, x:x + w]
            
            # Blend the overlay with the frame ROI
            cv2.addWeighted(overlay_resized, alpha, frame_roi, 1 - alpha, 0, frame_roi)
            
            # Place the updated ROI back into the frame
            image[y:y + h, x:x + w] = frame_roi

    return image

def run():
    global rect_start, rect_end, drawing, roi
    
    # Open camera feed
    cap = cv2.VideoCapture(0)
    
    # Create a window and set the mouse callback function
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_rectangle)
    
    # Initialize variables
    roi = None
    template = None
    template_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Draw the rectangle and mesh overlay on the frame
        if rect_start and rect_end:
            if roi is None:
                x1, y1, x2, y2 = extract_roi(frame, rect_start, rect_end)
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    template = roi
                    if template is not None and template.size > 0:
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    else:
                        print("ROI extraction failed, template is empty")
                else:
                    print("Invalid ROI coordinates")
            
            # Check if template is valid before using it
            if template is not None and template_gray is not None:
                # Convert current frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Template matching
                res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # Check if the matching value is above a threshold
                threshold = 0.8  # Adjust the threshold as needed
                if max_val >= threshold:
                    top_left = max_loc
                    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                    # Draw the rectangle where the template was found
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    # Draw the mesh on the matched area
#                    frame = draw_mesh(frame, top_left, bottom_right)
                    frame = overlay_image_roi(frame, top_left, bottom_right)
                else:
                    print("Area not found")
            else:
                print("Template or template_gray is None")
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()
