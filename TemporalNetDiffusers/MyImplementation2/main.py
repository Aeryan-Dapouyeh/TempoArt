from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import argparse
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision.io.video import read_video, write_video
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from controlnet_aux import HEDdetector


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-i', '--inputDir', help='Path to the input file', required=True)
parser.add_argument('-p', '--prompt', help='Prompt to the SD model', required=True)
parser.add_argument('-o', '--outputDir', help='Path to the output file', default=None)


# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
input_video_path = args.inputDir
prompt = args.prompt
outputDir = args.outputDir


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


input_video, _, info = read_video(input_video_path, pts_unit="sec", output_format="TCHW")
initialframe = input_video[0]
image = initialframe
image = image.permute(1, 2, 0)


hed = HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir="/work3/s204158/HF_cache")
image_HED = hed(image)


initial_image_transformed = hed_pipe(prompt, image_HED, num_inference_steps=20).images[0]

output_video = [initial_image_transformed]

for i in range(1, len(input_video)):
  curr = input_video[i]
  # curr = curr.permute(1, 2, 0)
  curr = F.to_pil_image(curr)
  curr = curr.resize(initial_image_transformed.size)

  prev = input_video[i-1]
  prev_transformed = output_video[-1]
  
  outputImg = pipe([prompt]*2, [curr, prev_transformed], num_inference_steps=20).images[0]
  output_video.append(outputImg)

output_video_pytorch = []

for frame in output_video:
  img = frame
  img_array = np.array(img)
  tensor = torch.from_numpy(img_array)
  output_video_pytorch.append(tensor)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video(outputDir, output_video_pytorch, fps=info["video_fps"])
