import os
import cv2

directory = os.getcwd()

for filename in os.listdir(directory):
    
    if filename.endswith('.mp4'):
        # Do something with the file
        print(filename)
        
        cap = cv2.VideoCapture(filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('{}_resized.mp4'.format(filename[:-4]), fourcc, 30, (512, 512))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (512, 512))
            out.write(frame_resized)
            # Do something with the resized frame
        
        cap.release()
        out.release()

