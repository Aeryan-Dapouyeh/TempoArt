import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)


## TODO: Wandb

### Dataset and dataloader
### To be done

### We assume there is a folder called Dataset
### Where there is a a folder for each datapoint
### The name of each datapoint folder is an index, e.g. 381 or 54325 
### Each datapoint folder contains 
### 1. The original frames F1 and F2, named F1.png and F2.png
### 2. Optical flow map of F1 and F2, named oF.png
### 3. A text file with a prompt describing the images, named prompt.txt
class TextualInversionDataset(Dataset):
    def __init__(
            self,
            DataDir,
            sizeThreshhold = 512 # Height or width is bigger then divide the picture until its short enough
    ):
        self.DataDir = DataDir

    def __len__(self):
        path = self.DataDir
        dir_count = sum(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
        return dir_count
    
    def __getitem__(self, i):
        currentDir = self.DataDir.path.join("{}".format(i))
        F1_path = os.path.join(currentDir, "F1.png")
        F2_path = os.path.join(currentDir, "F2.png")
        Of_path = os.path.join(currentDir, "Of.png")
        prompt_path = os.path.join(currentDir, "prompt.txt")

        prompt = ""

        with open(prompt_path, 'r') as f:
            prompt = f.read()

        F1 = read_image(F1_path)
        F2 = read_image(F2_path)
        Of = read_image(Of_path)

        return F1, F2, Of, prompt

# TODO: Add more necassary arguments according to the original train script
# e.g. learnable property should be style

DataDirectory = os.path.join(os.getcwd(),"Dataset")

train_dataset = TextualInversionDataset(DataDir=DataDirectory)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True #, num_workers=args.dataloader_num_workers
)

### Model
### TODO: Should be a hed controlnet

model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet"
)


### Loss function
loss_fn = torch.nn.MSELoss()

### Optimizer
optimizer = optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

### Training loop

for epoch in range(0, 50):
    text_encoder.train()
    ### See line 835 for the rest
    ### https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py#L573

print("Done!")
