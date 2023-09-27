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


@torch.inference_mode()
def stylize_video(
    input_video: Tensor,
    prompt: str,
    strength: float = 0.7,
    num_steps: int = 20,
    guidance_scale: float = 20, # 7.5,
    controlnet_scale: float = 0.5, # 1,
    batch_size: int = 4,
    height: int = 512,
    width: int = 512,
    low_threshold: int = 100,
    high_threshold: int = 200,
    device: str = "cuda",
) -> Tensor:
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, 
        cache_dir="/work3/s204158/HF_cache"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    prompt = prompt


    output_video = []
    generator = torch.Generator(device="cuda").manual_seed(42)

    for i in range(0, len(input_video)):
        image = input_video[i]
        image = image.permute(1, 2, 0)

        image = np.array(image)

        low_threshold = low_threshold
        high_threshold = high_threshold

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        output = pipe(
            prompt,
            canny_image,
            negative_prompt="monochrome, lowres, bad anatomy",
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
    return output_video_pytorch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=stylize_video.__doc__)
    parser.add_argument("-i", "--in-file", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o", "--out-file", type=str, default=None)
    parser.add_argument("-s", "--strength", type=float, default=0.7)
    parser.add_argument("-S", "--num-steps", type=int, default=20)
    parser.add_argument("-g", "--guidance-scale", type=float, default=7.5)
    parser.add_argument("-c", "--controlnet-scale", type=float, default=1.0)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-H", "--height", type=int, default=512)
    parser.add_argument("-W", "--width", type=int, default=512)
    parser.add_argument("-LT", "--LowerT", type=int, default=100)
    parser.add_argument("-HT", "--HigherT", type=int, default=200)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    input_video, _, info = read_video(args.in_file, pts_unit="sec", output_format="TCHW")
    # input_video = input_video.div(255)

    output_video = stylize_video(
        input_video=input_video,
        prompt=args.prompt,
        strength=args.strength,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        controlnet_scale=args.controlnet_scale,
        height=args.height,
        width=args.width,
        device=args.device,
        low_threshold = args.LowerT,
        high_threshold = args.HigherT,
        batch_size=args.batch_size,
    )

    # out_file = f"{Path(args.in_file).stem} | {args.prompt}.mp4" if args.out_file is None else args.out_file
    if args.out_file is None: 
        out_file = "{}.mp4".format(str(args.prompt))
    else: 
        # The name of the output file must end with mp4
        out_file = args.out_file
    # out_file = "Cat2.mp4"
    write_video(out_file, output_video.mul(255), fps=info["video_fps"])


'''
Some prompts
python StylizeAvideo.py -i "Input/AfricaDrone.mp4" -p "In Van gogh style, detailed" -o "AfricaDrone_vanGogh.mp4"
python StylizeAvideo.py -i "Input/BloodSweatandTears.mp4" -p "Men dancing, In Van gogh style, detailed, Van gogh painting" -o "BloodSweatandTears_vanGogh.mp4"
python StylizeAvideo.py -i "Input/Buckbeak.mp4" -p "Harry potter standing, In Van gogh style, detailed, Van gogh painting" -o "Buckbeak_vanGogh.mp4"
python StylizeAvideo.py -i "Input/DeathSceneTitanic.mp4" -p "A man and a woman besides eachother, dark blue, In Van gogh style, detailed, Van gogh painting" -o "DeathSceneTitanic.mp4"
python StylizeAvideo.py -i "Input/MidsommarDanceClip.mp4" -p "Woemn dancing in white dresses, In Van gogh style, detailed" -o "MidsommarDanceClip_vanGogh.mp4"
'''

