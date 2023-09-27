import argparse
import warnings
from pathlib import Path

import torch
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from torch import Tensor
from torchvision.io.video import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import trange

torch.cuda.empty_cache()
raft_transform = Raft_Large_Weights.DEFAULT.transforms()


@torch.inference_mode()
def stylize_video(
    input_video: Tensor,
    prompt: str,
    strength: float = 0.7,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 1,
    batch_size: int = 4,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
) -> Tensor:
    """
    Stylize a video with temporal coherence (less flickering!) using HuggingFace's Stable Diffusion ControlNet pipeline.

    Args:
        input_video (Tensor): Input video tensor of shape (T, C, H, W) and range [0, 1].
        prompt (str): Text prompt to condition the diffusion process.
        strength (float, optional): How heavily stylization affects the image.
        num_steps (int, optional): Number of diffusion steps (tradeoff between quality and speed).
        guidance_scale (float, optional): Scale of the text guidance loss (how closely to adhere to text prompt).
        controlnet_scale (float, optional): Scale of the ControlNet conditioning (strength of temporal coherence).
        batch_size (int, optional): Number of frames to diffuse at once (faster but more memory intensive).
        height (int, optional): Height of the output video.
        width (int, optional): Width of the output video.
        device (str, optional): Device to run stylization process on.

    Returns:
        Tensor: Output video tensor of shape (T, C, H, W) and range [0, 1].
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence annoying TypedStorage warnings

        '''
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=ControlNetModel.from_pretrained("wav/TemporalNet2", torch_dtype=torch.float16),
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir="/work3/s204158/HF_cache"
        ).to(device) '''
        
        
        pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("radames/stable-diffusion-v1-5-img2img", torch_dtype=torch.float16, 
            safety_checker=None,
            cache_dir="/work3/s204158/HF_cache").to("cuda")


        pipe_Hed = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "radames/stable-diffusion-v1-5-img2img",
            controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16),
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir="/work3/s204158/HF_cache"
        ).to(device)

        pipe_TemporalNet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "radames/stable-diffusion-v1-5-img2img",
            controlnet=ControlNetModel.from_pretrained("CiaraRowles/TemporalNet", torch_dtype=torch.float16),
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir="/work3/s204158/HF_cache"
        ).to(device)
        
        
        pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
        pipe_img2img.enable_xformers_memory_efficient_attention()
        pipe_img2img._progress_bar_config = dict(disable=True)

        pipe_Hed.scheduler = DPMSolverMultistepScheduler.from_config(pipe_Hed.scheduler.config)
        pipe_Hed.enable_xformers_memory_efficient_attention()
        pipe_Hed._progress_bar_config = dict(disable=True)

        pipe_TemporalNet.scheduler = DPMSolverMultistepScheduler.from_config(pipe_TemporalNet.scheduler.config)
        pipe_TemporalNet.enable_xformers_memory_efficient_attention()
        pipe_TemporalNet._progress_bar_config = dict(disable=True)


    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval().to(device)

    print("The shape of the input video is {}.".format(input_video.shape))

    output_video = []
    for i in trange(1, len(input_video), batch_size, desc="Diffusing...", unit="frame", unit_scale=batch_size):
        prev = resize(input_video[i - 1 : i - 1 + batch_size], (height, width), antialias=True).to(device)
        curr = resize(input_video[i : i + batch_size], (height, width), antialias=True).to(device)
        prev = prev[: curr.shape[0]]  # make sure prev and curr have the same batch size (for the last batch)

        flow_img = flow_to_image(raft.forward(*raft_transform(prev, curr))[-1]).div(255)
        control_img = torch.cat((prev, flow_img), dim=1)

        # 1. Initial image + prompt using ControlNet Hed Boundary on the first frame of the video.

        output_hed, _ = pipe_Hed(
            prompt=[prompt] * prev.shape[0],
            image=prev,
            control_image=prev,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )
        print("The shape of the first hed pipeline is: {}.".format(output_hed.shape))

        # 2. img2img with the next unaltered frame as the img2img input, and two ControlNet modules together, hed with the previously mentioned unaltered frame and the 
        # result of step 1 fed into the TemporalNet module
        
        output_img2img, _ = pipe_img2img(
            prompt=[prompt] * curr.shape[0],
            image=curr,
            # control_image=control_img,
            # height=height,
            # width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            # controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )

        print("The shape of the first image2image pipeline is: {}.".format(output_img2img.shape))
        print("The shape of the control image is: {}.".format(control_img.shape))

        output_hed2, _ = pipe_Hed(
            prompt=[prompt] * output_img2img.shape[0],
            image=output_img2img,
            control_image=flow_img,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )

        print("The shape of the first hed 2 pipeline is: {}.".format(output_hed2.shape))
        
        output_tempNet, _ = pipe_TemporalNet(
            prompt=[prompt] * output_hed.shape[0],
            image=output_hed,
            control_image=output_hed2,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )

        # 3. put the results of that into the the next frame's TemporalNet settings and repeat for the rest of the frames.
        
        output_tempNet2, _ = pipe_TemporalNet(
            prompt=[prompt] * curr.shape[0],
            image=curr,
            control_image=output_tempNet,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )


        output_video.append(output_tempNet2.permute(0, 2, 3, 1).cpu())
        
    return torch.cat(output_video)


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
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    input_video, _, info = read_video(args.in_file, pts_unit="sec", output_format="TCHW")
    input_video = input_video.div(255)

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
