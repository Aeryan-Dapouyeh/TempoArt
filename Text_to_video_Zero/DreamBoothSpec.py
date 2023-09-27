from huggingface_hub import hf_hub_download
from PIL import Image
import imageio
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor


# Download a demo video
filename = "__assets__/canny_videos_mp4/girl_turning.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

# Read video from path
reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
canny_edges = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]


# set model id to custom model
model_id = "PAIR/text2video-zero-controlnet-canny-avatar"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)
# latents = torch.randn((1, 4, 270, 270), device="cuda", dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)

prompt = "oil painting of a beautiful girl avatar style"
result = pipe(prompt=[prompt] * len(canny_edges), image=canny_edges, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
