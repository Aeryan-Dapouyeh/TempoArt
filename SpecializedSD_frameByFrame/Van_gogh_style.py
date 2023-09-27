import argparse
import warnings

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from tqdm import trange
from torch import Tensor
from torchvision.transforms.functional import resize
from torchvision.io.video import read_video, write_video
import torch



pipe = StableDiffusionPipeline.from_pretrained("dallinmackay/Van-Gogh-diffusion", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe._progress_bar_config = dict(disable=True)


prompt = "lvngvncnt, beautiful woman at sunset"
image = pipe(prompt).images[0]

@torch.inference_mode()
def stylize_video(
    input_video: Tensor,
    prompt: str,
    strength: float = 0.7,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 0.5, # 1,
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

        '''pipe = StableDiffusionImg2ImgPipeline.from_pretrained("dallinmackay/Van-Gogh-diffusion", torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe._progress_bar_config = dict(disable=True)'''

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "dallinmackay/Van-Gogh-diffusiong",
            controlnet=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir="/work3/s204158/HF_cache"
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe._progress_bar_config = dict(disable=True)


    print("The shape of the input video is {}.".format(input_video.shape))

    output_video = []
    for i in trange(1, len(input_video), batch_size, desc="Diffusing...", unit="frame", unit_scale=batch_size):
        curr = resize(input_video[i : i + batch_size], (height, width), antialias=True).to(device)

        output, _ = pipe(
            prompt=[prompt] * curr.shape[0],
            image=curr,
            # height=height,
            # width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            return_dict=False,
        )

        output_video.append(output.permute(0, 2, 3, 1).cpu())

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

