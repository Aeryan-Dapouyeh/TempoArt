# Importing all necessary libraries
import cv2
import os
#from torchvision.io.video import read_video, write_video
  
# Put in the video name

def getFram(vidname, output_dir):


    # Read the video from specified path
    cam = cv2.VideoCapture(os.path.join("movieClips", f"{vidname}.mp4"))
    #cam, _, info = read_video(os.path.join("movieClips", f"{vidname}.mp4"), pts_unit="sec", output_format="TCHW")
    

    # Input the name of the folder where you want the data to be stored
    dirname = output_dir

    savePath = os.path.join(".", f"{dirname}")

    try:

        # creating a folder named data
        if not os.path.exists(savePath):
            os.makedirs(dirname)

    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    
    while(True):

        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = f'./{dirname}/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
        
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()