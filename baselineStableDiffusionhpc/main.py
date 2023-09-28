import os
import cv2

from getFrames import getFram
from videoConnect import connectVideo
from SDpipe import SDPipeline

######################## Frame extraction #########################
list_of_prompts = ["Van gogh style painting of a male dancer dancing, detailed","Van gogh style painting of a man dancing, detailed", "Van gogh style painting of a black man laughing, detailed", "Van gogh style painting of a man with a headphone, detailed", "Van gogh style painting of woman running, detailed", "Van gogh style painting of woman running in the beach, detailed", "Van gogh style painting of a woman smiling, detailed", "Van gogh style painting of a woman, detailed", "Van gogh style painting of a woman working with her computer, detailed", "Van gogh style painting of a man walking in the desert, detailed"]

directory = 'movieClips'

for idx, filename in enumerate(sorted(os.listdir(directory))):
    f = os.path.join(directory, filename)
    
    if os.path.isfile(f):
        namevid = os.path.basename(f).split('.')[0]

        frame_out = f"{namevid}Untouched"
        getFram(namevid, frame_out)



############################# Pipeline #########################

        #print("Please enter the name of the directory where you keep the untouched image files:")
        #inp_dir = str(input())

        #print("Please enter the directory where you would like to store the retouched image files:")
        out_dir = f"{namevid}Retouched"

        #print("Please enter the prompt you wish to use:")
        prompt = list_of_prompts[idx]
        #print("Please enter the prompt or prompts to guide what to not include in image generation:")
        n_prompt = "bad anatomy, ugly, blurry, deformed, bad, cut-off, spotty, overlapping"

        SDPipeline(frame_out, out_dir, prompt, n_prompt)

    # def SDPipeline(type_of_diff, input_dir, output_dir, prompt, n_prompt):

    # negative_prompt (str or List[str], optional) â€” The prompt or prompts to guide what to 
    # not include in image generation. If not defined, you need to pass negative_prompt_embeds instead.

############################# Video connection #########################

        cap = cv2.VideoCapture(f)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        #print("Please enter the name of the video you wish to create:")
        vid_name = f"{namevid}Retouchedvid"

        connectVideo(out_dir, vid_name, fps)
        

for vid in os.listdir(os.getcwd()):
    if vid.endswith(".mp4"):
        src_path = os.path.join(os.getcwd, vid)
        dst_path = os.path.join(os.path.join(".", "final_videos"), vid)
        os.rename(src_path, dst_path)

