import torch
import os
import random
from PIL import Image
import numpy as np
from torchvision.io.video import write_video
from torchvision.transforms.functional import pil_to_tensor
from controlnet_aux import HEDdetector
from torch.utils.data import Dataset
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




class ControlNetDataSet(Dataset):
    def __init__(
        self,
        data_root,
        size=512,
        repeats=100,
        flip_p=0.5,
        center_crop=False,
        resolution = 512
    ):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.resolution = resolution

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = Image.Resampling.BILINEAR
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        self.image_transforms = transforms.Compose(
        [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        path = self.data_root
        dir_count = sum(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
        return dir_count
    

    def preprocessImage(self, ImgPath, Image_transforms):
        image = Image.open(ImgPath)
        image.convert("RGB")
        image = Image_transforms(image)

        return image
    
    
    def __getitem__(self, i):
        currentDir = os.path.join(self.data_root, "{}".format(i))
        F1_path = os.path.join(currentDir, "F1.png")
        F2_path = os.path.join(currentDir, "F2.png")
        Of_path = os.path.join(currentDir, "Of.png")
        F1_Styled_path = os.path.join(currentDir, "F1_Styled.png")
        prompt_path = os.path.join(currentDir, "prompt.txt")

        Raw_prompt = " "

        with open(prompt_path, 'r') as f:
            Raw_prompt = f.read()

        Raw_prompt = f"Van gogh style painting of {Raw_prompt} masterpiece"


        F1 = self.preprocessImage(ImgPath=F1_path, Image_transforms=self.conditioning_image_transforms)
        F2 = self.preprocessImage(ImgPath=F2_path, Image_transforms=self.image_transforms)
        F1_Styled = self.preprocessImage(ImgPath=F1_Styled_path, Image_transforms=self.image_transforms)
        Of = self.preprocessImage(ImgPath=Of_path, Image_transforms=self.image_transforms)

        return F1, F2, Of, F1_Styled, Raw_prompt


validation_dataset = ControlNetDataSet(
    data_root=validationSet_path,
    size=512,
)

validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=False
)



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


### Generate the rest of the video in a loop

for index, batch in enumerate(validation_dataloader):
    F1, F2, Of, F1_Styled, Raw_prompt = batch
    with torch.autocast("cuda"):
        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ]
        )

        Styled_F1 = temporalOutput[-1]

        Styled_F1.convert("RGB")
        Styled_F1 = conditioning_image_transforms(Styled_F1)

        # Styled_F1 = np.array(Styled_F1)
        # Styled_F1 = torch.from_numpy(Styled_F1).permute(2, 1, 0)
        Styled_F1 = torch.unsqueeze(Styled_F1, 0)

        image = TempoArt_pipeline(
            Raw_prompt[0], image=Styled_F1, control_image=torch.concatenate((Styled_F1, Of), dim=1), num_inference_steps=20, generator=generator
        ).images[0]

        hedImage = pil_to_tensor(image).permute(1, 2, 0)
        hedImage = hed(hedImage)

        image = CotrolNet_pipeline(prompt=Raw_prompt[0], image=hedImage, num_inference_steps=20).images[0]
        ### Used to be image=F1, control_iamge=conc(F2, Of)
        temporalOutput.append(image)


### Convert the generated images to a video
output_video_pytorch = []

for img in temporalOutput:
    img = np.array(img)
    img = torch.from_numpy(img)
    output_video_pytorch.append(img)

output_video_pytorch = torch.stack(output_video_pytorch)
write_video("StyledVid.mp4", output_video_pytorch, fps=24)