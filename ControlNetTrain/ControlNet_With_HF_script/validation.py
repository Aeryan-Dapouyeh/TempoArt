import torch
import os
from PIL import Image
import numpy as np
from torchvision.io.video import write_video
from torchvision.transforms.functional import pil_to_tensor
from controlnet_aux import HEDdetector
from torchvision import transforms
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline
)
from diffusers.pipelines.controlnet.costume_pipeline_controlnet_img2img import CostumeStableDiffusionControlNetImg2ImgPipeline
from costumeControlNet import CostumeControlNetModel


temporArt_path = "/work3/s204158/ControlNetTrain_With_originalScript/output/Giddy_frost_34_85000/controlnet"
validationSet_path = "/work3/s204158/VideoDataSet/ValidationDataset"

### Create the hed pipeline
hed = HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir = "/work3/s204158/HF_cache")

Hed_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", cache_dir = "/work3/s204158/HF_cache")
CotrolNet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=Hed_controlnet, cache_dir = "/work3/s204158/HF_cache").to("cuda")

CotrolNet_pipeline.scheduler = UniPCMultistepScheduler.from_config(CotrolNet_pipeline.scheduler.config)
CotrolNet_pipeline.set_progress_bar_config(disable=True)

### Create the tempoart pipeline
TempoArt_controlnet = CostumeControlNetModel.from_pretrained(temporArt_path, cache_dir = "/work3/s204158/HF_cache")
TempoArt_pipeline = CostumeStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-2-1-base", controlnet=TempoArt_controlnet, cache_dir = "/work3/s204158/HF_cache").to("cuda")

TempoArt_pipeline.scheduler = UniPCMultistepScheduler.from_config(TempoArt_pipeline.scheduler.config)
TempoArt_pipeline.set_progress_bar_config(disable=True)

num_Images = sum(os.path.isdir(os.path.join(validationSet_path, f)) for f in os.listdir(validationSet_path))


### Convert the initial image to the desired style
initialImage_path = os.path.join(validationSet_path, f"0/F1.png")
initialImagePrompt_path = os.path.join(validationSet_path, f"0/prompt.txt")
initialImagePrompt = ""

with open(initialImagePrompt_path, 'r') as f:
    initialImagePrompt = f.read()

initialImage = pil_to_tensor(Image.open(initialImage_path)).permute(1, 2, 0)
initialImage_hed = hed(initialImage)


initialImagePrompt = f"Van gogh style painting of {initialImagePrompt}, masterpiece"
initialImage_styled = CotrolNet_pipeline(prompt=initialImagePrompt, image=initialImage_hed, num_inference_steps=20).images[0]


initialImage_styled.save("Init.png")

temporalOutput = [initialImage_styled]

generator = torch.Generator(device="cuda").manual_seed(42)


for i in range(1, num_Images):
    currentDir = os.path.join(validationSet_path, "{}".format(i))
    F1_path = os.path.join(currentDir, "F1.png")
    F2_path = os.path.join(currentDir, "F2.png")
    Of_path = os.path.join(currentDir, "Of.png")
    prompt_path = os.path.join(currentDir, "prompt.txt")

    StyledImg = temporalOutput[-1]

    Raw_prompt = ""

    with open(prompt_path, 'r') as f:
        Raw_prompt = f.read()
    
    image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    F1 = pil_to_tensor(Image.open(F1_path)).permute(1, 2, 0)
    F1_hed = hed(F1)

    F1_Prompt = f"Van gogh style painting of {Raw_prompt}, masterpiece"
    
    F1_styled = CotrolNet_pipeline(prompt=F1_Prompt, image=initialImage_hed, num_inference_steps=20).images[0]
    F1_styled = pil_to_tensor(F1_styled)
    # F2 = pil_to_tensor(Image.open(F2_path)).permute(1, 2, 0)
    Of = pil_to_tensor(Image.open(Of_path))

    StyledImg = pil_to_tensor(StyledImg)

    # print(f"StyledImg has the shape {StyledImg.shape} and the dtype {StyledImg.dtype}.")
    # print(f"F1 has the shape {F1.shape} and the dtype {F1.dtype}.")
    # print(f"Of has the shape {Of.shape} and the dtype {Of.dtype}.")
    # print(f"Control image has the shape {torch.concatenate((F1, Of), dim=0).shape} and the dtype {torch.concatenate((F1, Of)).dtype}.")

    StyledImg = torch.unsqueeze(StyledImg, 0)
    controlImg = torch.unsqueeze(torch.concatenate((F1_styled, Of), dim=0), 0) 

    # Use F1_styled instead of StyledImg
    styledFrame = TempoArt_pipeline(Raw_prompt, image=F1_styled, control_image=controlImg, num_inference_steps=20, generator=generator).images[0]
    temporalOutput.append(styledFrame)



output_video_pytorch = []

for img in temporalOutput:
    img = np.array(img)
    img = torch.from_numpy(img)
    output_video_pytorch.append(img)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video("StyledVid.mp4", output_video_pytorch, fps=24)