import os
import cv2 
from PIL import Image


def connectVideo(input_dir, vid_name, fps):


    image_folder = os.path.join(".", input_dir)

    video_name = f'{vid_name}.mp4'

    
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
    


    # Array images should only consider
    # the image files ignoring others if any

    # Function to extract the index number from the file name
    def extract_idx(filename):
        try:
            return int(filename.split("frame")[1].split(".")[0])
        except ValueError:
            return float('inf')  # Return a large number for invalid filenames

    # Sort the list of image files based on the index number
    sorted_images = sorted(images)

    frame = cv2.imread(os.path.join(image_folder, sorted_images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  


    # const String &  	filename,   int  	fourcc,  double  	fps (Titanic is 48 fps),  Size  	frameSize,  bool  	isColor = true 
    video = cv2.VideoWriter(video_name, 0, fps, (width, height)) 

    # Appending the images to the video one by one
    for image in sorted_images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated


#connectVideo("BSVidRetouchedAnime", "BSVidRetouchedAnime", 25)



