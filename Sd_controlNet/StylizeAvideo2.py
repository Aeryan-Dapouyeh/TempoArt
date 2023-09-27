from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
import argparse
from torch import Tensor
from PIL import Image
import numpy as np
from torchvision.io.video import read_video, write_video
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


input_video, _, info = read_video("Input/BloodSweatandTears.mp4", pts_unit="sec", output_format="TCHW")

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
    cache_dir="/work3/s204158/HF_cache"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

prompt = "Van gogh painting of men dancing"
output_video = []
generator = torch.Generator(device="cuda").manual_seed(42)

for i in range(0, len(input_video)):
  image = input_video[i]
  image = image.permute(1, 2, 0)

  image = np.array(image)

  low_threshold = 100
  high_threshold = 200

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)

  output = pipe(
      prompt,
      canny_image,
      # negative_prompt="monochrome, lowres, bad anatomy",
      generator=generator,
      num_inference_steps=20,
  )
  output_video.append(output[0][0])

output_video_pytorch = []

for frame in output_video:
  img = frame
  img_array = np.array(img)
  tensor = torch.from_numpy(img_array)
  output_video_pytorch.append(tensor)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video("BloodSweatandTears_vanGogh.mp4", output_video_pytorch, fps=info["video_fps"])

