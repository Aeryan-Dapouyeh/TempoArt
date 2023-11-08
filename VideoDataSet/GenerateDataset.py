import os
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.io.video import read_video
from controlnet_aux import HEDdetector
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.models.optical_flow import raft_large
import torchvision.transforms as transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

torch.cuda.empty_cache()

# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16,
#     # cache_dir="/work3/s204158/HF_cache"
# )
# 
# generator = torch.Generator(device="cuda").manual_seed(42)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, generator=generator,
#     # cache_dir="/work3/s204158/HF_cache"
# ).to("cuda")
# 
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# 
# # Remove if you do not have xformers installed
# # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# # for installation instructions
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()
# 
# hed = HEDdetector.from_pretrained('lllyasviel/Annotators', 
#     # cache_dir="/work3/s204158/HF_cache"
#     )

def DownSample_Image(image):
    img = resize(image.permute(2, 1, 0), (512, 512))# .permute(1, 2, 0)
    img_PIL = to_pil_image(img)
    img_PIL = img_PIL.rotate(-90)
    return img.unsqueeze(0).float(), img_PIL

def list_directories(directory_path):
    filenames = os.listdir(directory_path)
    directories = [filename for filename in filenames if os.path.isdir(os.path.join(directory_path, filename))]
    return directories

def convert_and_find_max(strings_list):
    # Convert strings to integers
    integers_list = [int(num) for num in strings_list]

    # Find the highest integer
    highest_integer = max(integers_list)
    return highest_integer

def GenerateRandomDataset():
    root_directory = os.path.join(os.getcwd(), "newdata_Raw")
    ProcessedDir = os.path.join(os.getcwd(), "ProcessedVideo_Unstyled2")
    
    # OriginalVid = "C:\Users\arian\OneDrive\Desktop\Kunstig_Intelligens_og_Data\TempoArt\Input_baseline_images\Original\DYVid.mp4"
    Of_model = raft_large(pretrained=True, progress=False)
    Of_model = Of_model.eval()
    IndexCounter = 0

    VideoNames_list = []
    with open('Processed_videos.txt', 'r') as file:
        VideoNames_list = file.readlines()
        # Remove newline characters from each name
        VideoNames_list = [name.strip() for name in VideoNames_list]

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:

            if filename.lower().endswith(".mp4"):
                VideoPath = os.path.join(dirpath, filename)

                print(filename)
                input_video, _, info = read_video(VideoPath, pts_unit="sec", output_format="THWC")

                RepeatedFile = any(filename in item for item in VideoNames_list)
                if RepeatedFile:
                    continue


                F1 = input_video[0]

                Video_prompt = "{}, {}, {}"
                VideoDirName = filename[:-4]
                promptPath = os.path.join(root_directory, VideoDirName, "text.txt")
                

                with open(promptPath, 'r') as f:
                    content = f.read()
                    # stylePrompt = "Van gogh style painting of "
                    # additional_prompts = ", masterpiece"
                    # Video_prompt = Video_prompt.format(stylePrompt, content, additional_prompts)
                    Video_prompt = Video_prompt.format("", content, "")
                

                for i in range(1, len(input_video)):
                    pathName = os.path.join(ProcessedDir, "{}".format(IndexCounter))
                    # print(pathName)
                    # print("{}/{}".format(i, len(input_video)))

                    ### If there already is a datapoint with this number
                    ### Update the counter
                    pathExists = os.path.exists(pathName)
                    if pathExists:
                        AllPaths = list_directories(ProcessedDir)
                        MaxNumber = convert_and_find_max(AllPaths)
                        Possible_path = MaxNumber+1
                        pathName = os.path.join(ProcessedDir, "{}".format(Possible_path))
                        IndexCounter = Possible_path
                        
                        print("Path {} exists.".format(pathName))
                        print(f"The last path is {MaxNumber}. Can be replaced by {Possible_path}.")

                    os.makedirs(pathName, exist_ok=True)

                    F1Path = os.path.join(pathName, "F1.png")
                    F2Path = os.path.join(pathName, "F2.png")
                    StyledF1_path = os.path.join(pathName, "F1_Styled.png")
                    OfPath = os.path.join(pathName, "Of.png")
                    promptPath = os.path.join(pathName, "prompt.txt".format(IndexCounter))

                    F1_img = F1
                    F2 = input_video[i]
                    F2_img = F2

                    torch_F1_img, F1_img = DownSample_Image(F1_img)
                    torch_F2_img, F2_img = DownSample_Image(F2_img)

                    # Generate the styledImage
                    image = F1_img
                    
                    # Uncomment these lines for generating F1 styled
                    # image_HED = hed(image)
                    # image = pipe(Video_prompt, image_HED, num_inference_steps=20).images[0]
                    image = torch.randn([3, 512, 512])
                    F1Styled_img = image
                    
                    # Generate random images as Of, for now
                    # Of_random = torch.randn([3, 512, 512])
                    Of = Of_model(torch_F1_img, torch_F2_img)[-1]

                    # Save the image
        
                    F1_img.save(F1Path)
                    F2_img.save(F2Path)
                    # F1Styled_img.save(StyledF1_path)
                    save_image(F1Styled_img, StyledF1_path)
                    save_image(Of, OfPath)
                    # Of_random.save(Of_random)

                    with open(promptPath, 'w') as f:
                        f.write(Video_prompt)

                    F1 = input_video[i]

                    IndexCounter += 1
                VideoNames_list.append(filename)
            
        # Write the name of the videos
        with open("Processed_videos.txt", 'w') as file:
            for VideoName in VideoNames_list:
                # Write each name followed by a newline character
                file.write(f"{VideoName}\n")

if __name__ == '__main__':
    GenerateRandomDataset()