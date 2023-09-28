# Import packages
import torch
torch.cuda.empty_cache()
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from PIL import Image
import os


# Create the pipeline
#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

def SDPipeline(input_dir, output_dir, prompt, n_prompt):

    device = "cuda"

    # Different types of pipelines: img2img and depth2img
    
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth",torch_dtype=torch.float16, cache_dir="/work3/s204134/HF_cache")
    pipe.safety_checker = None
    
    pipe = pipe.to(device)


    image_dir = os.path.join(".", f"{input_dir}")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]


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




    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        init_image = Image.open(image_path)
        init_image.resize((896, 512))

        output_filename = f"reimaginedframe{idx}.png"
        output_path = os.path.join(savePath, output_filename)
        
        
        image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.7)[0]
        image[0].save(output_path)
            