# Import packages
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from transformers import pipeline
from PIL import Image
import os
from resizeImg import resize_img
from torchvision import transforms


# Create the pipeline
#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

def SDPipelineHED(input_dir, output_dir, prompt, n_prompt):

    device = "cuda"
    
    transform = transforms.Compose([transforms.PILToTensor()]) 

    # Different types of pipelines: img2img and depth2img
    
    hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
    
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)

    #pipe = StableDiffusionImg2ImgPipeline.from_pretrained(hyperparameters["output_dir"], scheduler=DPMSolverMultistepScheduler.from_pretrained(hyperparameters["output_dir"], subfolder="scheduler"), torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.safety_checker = None
    pipe = pipe.to(device)
    

    image_dir = os.path.join(".", f"{input_dir}")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    #image_files = [hed(f.permute(1,2,0)) for f in image_files_original]



    # Function to extract the index number from the file name
    def extract_idx(filename):
        try:
            return int(filename.split("frame")[1].split(".")[0])
        except ValueError:
            return float('inf')  # Return a large number for invalid filenames

    # Sort the list of image files based on the index number
    sorted_image_files_original = sorted(image_files, key=extract_idx)

    sorted_images_files_tensor = [transform(Image.open(os.path.join(image_dir, f))) for f in sorted_image_files_original]

    #sorted_image_files = [hed(f.permute(1,2,0)) for f in sorted_images_files_tensor]

    # Input the name of the folder where you want the data to be stored
    #relativePath = f"./baselineStableDiffusion/{dirname}/"
    #savePath =  os.path.abspath(relativePath)

    #savePath = f"./baselineStableDiffusion/{dirname}/"
    savePath = os.path.join(".", output_dir)


    try:

        # creating a folder named data
        if not os.path.exists(savePath):
            #print(os.path.exists(savePath))
            os.makedirs(output_dir, exist_ok=True)
            print(os.path.exists(savePath))


    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
        print(os.path.exists(savePath))




    for idx, image_file in enumerate(sorted_images_files_tensor):
        image = image_file.permute(1,2,0)
        init_image = hed(image)

        output_filename = f"{str(idx).zfill(5)}.png"
        output_path = os.path.join(savePath, output_filename)
        
        
        images = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt).images
        images[0].save(output_path)


