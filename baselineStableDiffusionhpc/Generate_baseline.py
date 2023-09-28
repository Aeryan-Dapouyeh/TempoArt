# Import packages
import torch
torch.cuda.empty_cache()

from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline, UniPCMultistepScheduler
from PIL import Image
import argparse
from torchvision.io.video import read_video, write_video
import torchvision.transforms.functional as F
import numpy as np


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-i', '--inputDir', help='Path to the input file', required=True)
parser.add_argument('-p', '--prompt', help='Prompt to the SD model', required=True)
parser.add_argument('-o', '--outputDir', help='Path to the output file', default=None)
parser.add_argument('-o', '--cacheDir', help='Path to the cache directory in HPC', default="/work3/s204158/HF_cache")


# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
input_dir = args.inputDir
prompt = args.prompt
output_dir = args.outputDir
cacheDir = args.cacheDir


pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16,
    cache_dir=cacheDir
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

input_video, _, info = read_video(input_dir, pts_unit="sec", output_format="TCHW")

output_video = []

for i in range(0, len(input_video)):
  curr = input_video[i]
  curr = F.to_pil_image(curr)
  curr = curr.resize((512, 512))


  
  outputImg = pipe(prompt, curr, num_inference_steps=20).images[0]
  output_video.append(outputImg)


output_video_pytorch = []

for frame in output_video:
  img = frame
  img_array = np.array(img)
  tensor = torch.from_numpy(img_array)
  output_video_pytorch.append(tensor)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video(output_dir, output_video_pytorch, fps=info["video_fps"])    