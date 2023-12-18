import os
import cv2

from getFrames import getFram
from videoConnect import connectVideo
from resizeImg import resize_img
from SDpipeHED import SDPipelineHED

#"Van Gogh style painting of a male dancer dancing, detailed",
#                       "Van Gogh style painting of a man dancing, detailed", 
#                       "Van Gogh style painting of a black man laughing, detailed", 
#                       "Van Gogh style painting of a man with a headphone, detailed",

######################## Frame extraction #########################
prompt_di = {"vanlist": ["Van Gogh style painting of a male dancer dancing, detailed",
                       "Van Gogh style painting of a man dancing, detailed", 
                       "Van Gogh style painting of a black man laughing, detailed", 
                       "Van Gogh style painting of a man with a headphone, detailed",
                       "Van Gogh style painting of woman running, detailed", 
                       "Van Gogh style painting of woman running on the beach with a beachball, detailed",
                       "Van Gogh style painting of a woman smiling, detailed", 
                       "Van Gogh style painting of a woman, detailed", 
                       "Van Gogh style painting of a woman working with her computer, detailed", 
                       "Van Gogh style painting of a man walking in the desert, detailed"],
            "anilist": ["Anime style drawing of a male dancer dancing, detailed",
                       "Anime style drawing of a man dancing, detailed", 
                       "Anime style drawing of a black man laughing, detailed", 
                       "Anime style drawing of a man with a headphone, detailed", 
                       "Anime style drawing of woman running, detailed", 
                       "Anime style drawing of woman running on the beach with a beachball, detailed",
                       "Anime style drawing of a woman smiling, detailed", 
                       "Anime style drawing of a woman, detailed", 
                       "Anime style drawing of a woman working with her computer, detailed", 
                       "Anime style drawing of a man walking in the desert, detailed"],
            "monetlist": ["Monet style painting of a male dancer dancing, detailed",
                       "Monet style painting of a man dancing, detailed", 
                       "Monet style painting of a black man laughing, detailed", 
                       "Monet style painting of a man with a headphone, detailed", 
                       "Monet style painting of woman running, detailed", 
                       "Monet style painting of woman running on the beach with a beachball, detailed",
                       "Monet style painting of a woman smiling, detailed", 
                       "Monet style painting of a woman, detailed", 
                       "Monet style painting of a woman working with her computer, detailed", 
                       "Monet style painting of a man walking in the desert, detailed"]

}


print("Please enter the type of prompts, you wish to use:")
print("options: van, anime, monet and base")
emb_type = str(input())

#emb_type = "anime"

if emb_type.lower() == "van":
    list_of_prompts = prompt_di["vanlist"]
elif emb_type.lower() == "anime":
    list_of_prompts = prompt_di["anilist"]
elif emb_type.lower() == "monet":
    list_of_prompts = prompt_di["monetlist"]




directory = 'movieClips'
for idx, filename in enumerate(sorted(os.listdir(directory))):
    f = os.path.join(directory, filename)
    
    if os.path.isfile(f):
        namevid = os.path.basename(f).split('.')[0]
        frame_out = f"{namevid}Untouched"

        #if os.listdir(frame_out):
        #    print("Your code ends here")
        #    continue
        #else:
        getFram(namevid, frame_out)
        resize_img(frame_out)

############################# Pipeline #########################

        #emb_type = "monet"
        #emb_type = "van"
        #emb_type = "anime"

        #print("Please enter the directory where you would like to store the retouched image files:")
        out_dir = f"{namevid}RetouchedHED{emb_type}"

        #print("Please enter the prompt you wish to use:")
        prompt = list_of_prompts[idx]
        print(prompt)

        #print("Please enter the prompt or prompts to guide what to not include in image generation:")
        n_prompt = "bad anatomy, ugly, blurry, deformed, bad, cut-off, spotty, overlapping"

        #SDPipeline("base", frame_out, out_dir, prompt, n_prompt)

        SDPipelineHED(frame_out, out_dir, prompt, n_prompt)
    # def SDPipeline(type_of_diff, input_dir, output_dir, prompt, n_prompt):

    # negative_prompt (str or List[str], optional) — The prompt or prompts to guide what to 
    # not include in image generation. If not defined, you need to pass negative_prompt_embeds instead.

############################# Video connection #########################

        cap = cv2.VideoCapture(f)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        #print("Please enter the name of the video you wish to create:")
        vid_name = f"{namevid}RetouchedHED{emb_type}"

        connectVideo(out_dir, vid_name, fps)
        

#for vid in os.listdir(os.getcwd()):
#    if vid.endswith(".mp4"):
#        src_path = os.path.join(os.getcwd, vid)
#        dst_path = os.path.join(os.path.join(".", "final_videos"), vid)
#        os.rename(src_path, dst_path)

