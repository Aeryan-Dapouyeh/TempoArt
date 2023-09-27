from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import argparse
import cv2
from PIL import Image
import numpy as np
from torchvision.io.video import read_video, write_video
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import cv2



# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-i', '--inputDir', help='Path to the input file', required=True)
parser.add_argument('-p', '--prompt', help='Prompt to the SD model', required=True)
parser.add_argument('-o', '--outputDir', help='Path to the output file', default=None)


# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
input_video = args.inputDir
prompt = args.prompt
outputDir = args.outputDir

'''
input_video = "Input/Buckbeak.mp4"
prompt = "Van gogh painting of harry potter standing, masterpiece"
outputDir = "Buckbeak_vanGogh.mp4"
'''


input_video, _, info = read_video(input_video, pts_unit="sec", output_format="TCHW")

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16,
    cache_dir="/work3/s204158/HF_cache"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16,
    cache_dir="/work3/s204158/HF_cache"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


output_video = []
generator = torch.Generator(device="cuda").manual_seed(42)

for i in range(0, len(input_video)):
  image = input_video[i]
  image = image.permute(1, 2, 0)
  hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
  image_HED = hed(image)
  image = pipe(prompt, image_HED, num_inference_steps=20).images[0]
  output_video.append(image)

output_video_pytorch = []

for frame in output_video:
  img = frame
  img_array = np.array(img)
  tensor = torch.from_numpy(img_array)
  output_video_pytorch.append(tensor)


if outputDir==None: 
  outputDir="{}.mp4".format(prompt)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video(outputDir, output_video_pytorch, fps=info["video_fps"])
