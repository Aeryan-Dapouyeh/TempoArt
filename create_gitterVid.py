import numpy as np
import cv2
import torchvision.io as io
import torch
import os
from videoConnect import connectVideo


def create_gitterVid(vid1, vid2, vid3, vid4, out_path, vid_name):
    """
    The vid1, vid2, vid3 and vid4 represent the paths to the videos to be stitched together
    """
    
    # Read the video from specified path
    vidframes1 = cv2.VideoCapture(vid1)
    print("vid1 has been loaded")
    vidframes2 = cv2.VideoCapture(vid2)
    print("vid2 has been loaded")
    vidframes3 = cv2.VideoCapture(vid3)
    print("vid3 has been loaded")
    vidframes4 = cv2.VideoCapture(vid4)
    print("vid4 has been loaded")
    
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print("Output directory already exists")
    
    #print("you made it here")
    
    desired_width = 520
    
    currentframe = 0
    
    while(True):
        
        ret1, frame1 = vidframes1.read()
        ret2, frame2 = vidframes2.read()
        ret3, frame3 = vidframes3.read()
        ret4, frame4 = vidframes4.read()
        
        if ret1:
        
            #frame1 = vidframes1[currentframe]
            #frame2 = vidframes2[currentframe]
            #frame3 = vidframes3[currentframe]
            #frame4 = vidframes4[currentframe]
            height, width = frame1.shape[:2]
            aspect_ratio = width / height
            new_height = int(desired_width / aspect_ratio)
        
        
            frame1 = cv2.resize(frame1, (desired_width, new_height))
            frame2 = cv2.resize(frame2, (desired_width, new_height))
            frame3 = cv2.resize(frame3, (desired_width, new_height))
            frame4 = cv2.resize(frame4, (desired_width, new_height))
            
            text1 = "Original"
            text2 = "Stable Diffusion"
            text3 = "HED-Controlnet"
            text4 = "Textual Inversion"
            
            text_position = (frame1.shape[1] - 200, frame1.shape[0] - 10) # Adjust these values as needed

            # Define font, scale, color, and thickness
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255) # White color
            line_type = 2

            #frame1 = cv2.copyMakeBorder(frame1, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
            #frame2 = cv2.copyMakeBorder(frame2, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
            #frame3 = cv2.copyMakeBorder(frame3, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            
            frame1 = cv2.putText(frame1, text1, text_position, font, font_scale, font_color, line_type)
            frame2 = cv2.putText(frame2, text2, text_position, font, font_scale, font_color, line_type)
            frame3 = cv2.putText(frame3, text3, text_position, font, font_scale, font_color, line_type)
            frame4 = cv2.putText(frame4, text4, text_position, font, font_scale, font_color, line_type)
            
            frame = cv2.vconcat([cv2.hconcat([frame1, frame2]), 
                                   cv2.hconcat([frame3, frame4])])

            #frame = np.concatenate((frame1, frame2, frame3, frame4), axis=1)

            frame_name = f"{out_path}/{str(currentframe).zfill(5)}.jpg"

            print("Creating..." + frame_name)

            cv2.imwrite(frame_name, frame)
            currentframe += 1
            
        else:
            break
        
    
    vidframes1.release()
    vidframes2.release()
    vidframes3.release()
    vidframes4.release()
    cv2.destroyAllWindows()
    
    cap = cv2.VideoCapture(vid1)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)

    connectVideo(out_path, vid_name, fps)

sp.eval(sp.exp(-1)-sp.exp(-sp.Rational(3,2)))



create_gitterVid("normalvid/DYVid.mp4","baselineSB/DYVidRetouched.avi", "SDHED/DYVidRetouchedHEDanime.mp4", "textualinversion/DYVidRetouched.avi", "gitterVid", "DYVidGitter")
