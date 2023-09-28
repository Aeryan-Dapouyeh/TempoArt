import os
import cv2 
from PIL import Image


def connectVideo(input_dir, vid_name, fps):
    
    image_folder = input_dir

    rerenderedPath = os.path.join(".", image_folder)

    video_name = f'{vid_name}.avi'

    
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
    
    # Array images should only consider
    # the image files ignoring others if any

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  


    # const String &  	filename,   int  	fourcc,  double  	fps (Titanic is 48 fps),  Size  	frameSize,  bool  	isColor = true 
    video = cv2.VideoWriter(video_name, 0, fps, (width, height)) 

    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated





