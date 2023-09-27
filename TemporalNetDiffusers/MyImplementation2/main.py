from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision.io.video import read_video, write_video
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from controlnet_aux import HEDdetector



torch.cuda.empty_cache()

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16
)

temporalNet = ControlNetModel.from_pretrained("CiaraRowles/TemporalNet", torch_dtype=torch.float16)

hed_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16,
    cache_dir="/work3/s204158/HF_cache"
)
hed_pipe.scheduler = UniPCMultistepScheduler.from_config(hed_pipe.scheduler.config)
hed_pipe.enable_xformers_memory_efficient_attention()
hed_pipe.enable_model_cpu_offload()

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=[controlnet, temporalNet], safety_checker=None, torch_dtype=torch.float16,
    cache_dir="/work3/s204158/HF_cache"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


input_video, _, info = read_video("Input/DYVid.mp4", pts_unit="sec", output_format="TCHW")
initialframe = input_video[0]
image = initialframe
image = image.permute(1, 2, 0)


hed = HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir="/work3/s204158/HF_cache")
image_HED = hed(image)


initial_image_transformed = hed_pipe("Van gogh painting of a man, masterpiece", image_HED, num_inference_steps=20).images[0]
initial_image_transformed


secondframe = input_video[1]
image2 = initial_image_transformed
#image2 = image2.permute(1, 2, 0)
# image2 = F.to_pil_image(image2)

image_HED2 = hed(image2)


print(image_HED2.size)
# Add a new dimension at the beginning of the tensor
#image2 = torch.unsqueeze(image2, 0)
print(image2.size)

image2_transformed = pipe(["Van gogh painting of a cat, masterpiece"]*2, [image2, initial_image_transformed], num_inference_steps=20).images[0]

image2_transformed.save("Frame2.png")
