# augmented-reality
Augment image on a selected area


'cd path/to/your/project'


// install requirements 

'pip install -r requirements.txt'


----->> run main.py with proper index <<--------

// run main file with parameter

'python3 main.py $index'

// used modules as per $index, for all module run() function is called

--> 0: will use simple canny detection to detect rectangle in frame and overlay a mesh

--> 1:  press enter to capture frame, and press enter again to continue, it will use ORB detector to match Keypoints from captured frame on the target frame. 

--> 2: use Yolo to detect flat surface and overlay image on it, in this case use 'cell phone'

--> 3.1: user can select a area on wall and image will be augmented over that row, using grey template matching

--> 3.3: user can select a area on wall and image will be augmented over that row, using grey cv2.TrackerCSRT_create
