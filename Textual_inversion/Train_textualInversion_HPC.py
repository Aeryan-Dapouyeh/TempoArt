import os
import sys
import numpy as np
import math
import safetensors
import logging
import torch
import diffusers
import shutil
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.models.optical_flow import raft_large
import transformers
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from controlnet_aux import HEDdetector
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    ControlNetModel, 
    StableDiffusionControlNetPipeline
)

## Config
train_batch_size = 2


## TODO: Wandb

logger = get_logger(__name__)

## TODO: Consider if the log_validation function in line 115 would be necassary

## TODO: We dont use args in this script so do something about it
def save_progress(text_encoder, placeholder_token_ids, accelerator, placeholder_token, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)




### Dataset and dataloader


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
            tokenizer,
            size=512,
            sizeThreshhold = 512 # TODO: Height or width is bigger then divide the picture until its short enough
    ):
        self.DataDir = DataDir
        self.size = size
        self.tokenizer = tokenizer

    def __len__(self):
        path = self.DataDir
        dir_count = sum(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
        return dir_count
    
    def preprocessImage(self, ImgPath):
        image = Image.open(ImgPath)

        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        
        crop = min(img.shape[0], img.shape[1])
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        
        image = image.resize((512, 512))

        flip_transform = transforms.RandomHorizontalFlip(p=0.5)

        image = flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image 

    
    def __getitem__(self, i):
        currentDir = os.path.join(self.DataDir, "{}".format(i))
        F1_path = os.path.join(currentDir, "F1.png")
        F2_path = os.path.join(currentDir, "F2.png")
        Of_path = os.path.join(currentDir, "Of.png")
        prompt_path = os.path.join(currentDir, "prompt.txt")

        Raw_prompt = ""

        with open(prompt_path, 'r') as f:
            Raw_prompt = f.read()
        ### COMMENT: OPTIMIZE THE PROMPT VECTOR INSTEAD 
        prompt = self.tokenizer(
            Raw_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # F1 = read_image(F1_path)
        # F2 = read_image(F2_path)
        # Of = read_image(Of_path)
        F1 = self.preprocessImage(ImgPath=F1_path)
        F2 = self.preprocessImage(ImgPath=F2_path)
        Of = self.preprocessImage(ImgPath=Of_path)

        return F1, F2, Of, prompt, Raw_prompt

# TODO: Add more necassary arguments according to the original train script
# e.g. learnable property should be style

output_dir = os.path.join(os.getcwd(), "output")
logging_dir = os.path.join(output_dir, "logging")

accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="no",
    # log_with="wandb", ## TODO: Uncomment this lise to get wandb to work
    project_config=accelerator_project_config,
)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

### TODO: it pains me to see this value not be 42
seed = np.int64(42)
set_seed(int(42))


### Model
### TODO: Should be a hed controlnet

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", cache_dir="/work3/s204158/HF_cache")
CotrolNet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet, cache_dir="/work3/s204158/HF_cache"
)
hed = HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir="/work3/s204158/HF_cache")

'''tokenizer = CotrolNet_pipeline.tokenizer

# Load scheduler and models
noise_scheduler = CotrolNet_pipeline.scheduler
text_encoder = CotrolNet_pipeline.text_encoder
vae = CotrolNet_pipeline.vae
unet = CotrolNet_pipeline.unet
'''


# model_id = "lllyasviel/sd-controlnet-hed"
model_id = "runwayml/stable-diffusion-v1-5"

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet = CotrolNet_pipeline.unet

placeholder_token = "<Spaghetti>"
placeholder_tokens = [placeholder_token]
# Number of embedding vectors to be used in the textual inversion model
num_vectors = 1


DataDirectory = os.path.join("/work3/s204158/TextualInv_Train","Dataset")

train_dataset = TextualInversionDataset(DataDir=DataDirectory, tokenizer=tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True #, num_workers=args.dataloader_num_workers
)

# add dummy tokens for multi-vector
additional_tokens = []
for i in range(1, num_vectors):
    additional_tokens.append(f"{placeholder_token}_{i}")
placeholder_tokens += additional_tokens

num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )


# Convert the initializer_token, placeholder_token to ids
initializer_token = "Monet"
token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

# Resize the token embeddings as we are adding new special tokens to the tokenizer
text_encoder.resize_token_embeddings(len(tokenizer))

# Initialise the newly added placeholder token with the embeddings of the initializer token
token_embeds = text_encoder.get_input_embeddings().weight.data
with torch.no_grad():
    for token_id in placeholder_token_ids:
        token_embeds[token_id] = token_embeds[initializer_token_id].clone()

# Freeze vae and unet
vae.requires_grad_(False)
unet.requires_grad_(False)
# Freeze all parameters except for the token embeddings in text encoder
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)


### TODO: Maybe consider if using this is beneficial
gradient_checkpointing = False
if gradient_checkpointing:
    # Keep unet in train mode if we are using gradient checkpointing to save memory.
    # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
    unet.train()
    text_encoder.gradient_checkpointing_enable()
    unet.enable_gradient_checkpointing()


enable_xformers_memory_efficient_attention = True
if enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

allow_tf32 = True
if allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

scale_lr = False
learning_rate = 1e-4
gradient_accumulation_steps = 1


if scale_lr:
    learning_rate = (
        learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
    )

### Loss function
loss_fn = torch.nn.MSELoss()

### Optimizer
optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
)

validation_steps = 100
max_train_steps = 500
num_train_epochs = 100

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = "constant"
lr_warmup_steps = 500
lr_num_cycles = 1

lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
    num_training_steps=max_train_steps * accelerator.num_processes,
    num_cycles=lr_num_cycles,
)


# Prepare everything with our `accelerator`.
text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    text_encoder, optimizer, train_dataloader, lr_scheduler
)

# For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
# Move vae and unet to device and cast to weight_dtype
unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
 
if accelerator.is_main_process:
    # TODO: Find a solution for args if results in a bug
    accelerator.init_trackers("textual_inversion")
    # accelerator.init_trackers("textual_inversion", config=vars(args))


# Train!
total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")
global_step = 0
first_epoch = 0


# Potentially load in the weights and states from a previous save
resume_from_checkpoint = False
if resume_from_checkpoint:
    if resume_from_checkpoint != "latest":
        path = os.path.basename(resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        accelerator.print(
            f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)
# keep original embeddings as reference
orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()


### Training loop

for epoch in range(first_epoch, num_train_epochs):
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(text_encoder):
            # Batch = F1, F2, OF, prompt, Raw_prompt
            # Size of the images are torch.Size([4, 3, 1024, 1024]), aka. they have the shape [b, c, H, W]
            # Fourth elemen in batch is a tuple of length b="batch_length" with b prompts in it 
            
            
            '''     
            # Convert images to latent space
            latents = vae.encode(batch[0].to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            '''

            # Get the text embedding for conditioning
            # encoder_hidden_states = text_encoder(batch[3])[0].to(dtype=weight_dtype).to("cuda")
            Rawprompts = batch[4]
    

            images_HED_F1 = [hed(batch[0][i].permute(1, 2, 0).cpu()) for i in range(batch[0].shape[0])]
            images_HED_F1 = [ToTensor()(image) for image in images_HED_F1]
            images_HED_F1 = torch.stack(images_HED_F1).to("cuda")

            images_HED_F2 = [hed(batch[1][i].permute(1, 2, 0).cpu()) for i in range(batch[1].shape[0])]
            images_HED_F2 = [ToTensor()(image) for image in images_HED_F2]
            images_HED_F2 = torch.stack(images_HED_F2).to("cuda")

            ### DIMITRI'S COMMENT: IT WON'T BE NECASSARY TO STYLE THE FIRST FRAME HERE, AND MAYBE EVEN NOT THE SECOND FRAME

            CotrolNet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
	            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, text_encoder=text_encoder, cache_dir="/work3/s204158/HF_cache").to("cuda")
            
            ### BIG TODO: Probably have to change num_inference_steps
            print(images_HED_F1.shape)
            # StyledImages_F1 = [CotrolNet_pipeline(prompt=[Rawprompts[i]]*train_batch_size, image=image, num_inference_steps=20).images[0] for i, image in enumerate(images_HED_F1)]
            # StyledImages_F2 = [CotrolNet_pipeline(prompt=[Rawprompts[i]]*train_batch_size, image=image, num_inference_steps=20).images[0] for i, image in enumerate(images_HED_F2)]
            StyledImages_F1 = CotrolNet_pipeline(prompt=[Rawprompts[0]]*train_batch_size, image=images_HED_F1, num_inference_steps=20).images[0]
            StyledImages_F2 = CotrolNet_pipeline(prompt=[Rawprompts[0]]*train_batch_size, image=images_HED_F2, num_inference_steps=20).images[0]

            StyledImages_F1 = [ToTensor()(image) for image in StyledImages_F1]
            StyledImages_F2 = [ToTensor()(image) for image in StyledImages_F2]
            
            StyledImages_F1 = torch.stack(StyledImages_F1).to("cuda")
            StyledImages_F2 = torch.stack(StyledImages_F2).to("cuda")

            # 1. Then generate the styled images for F0 too
            # 2. And generate Of maps from that
            # 3. The Of maps would be our modelPreds  

            # StyledImages_F1 = pil_to_tensor(StyledImages_F1)
            # StyledImages_F2 = pil_to_tensor(StyledImages_F2)

            # StyledImages_F1 = ToTensor()(StyledImages_F1).unsqueeze_(0).to("cuda")
            # StyledImages_F2 = ToTensor()(StyledImages_F2).unsqueeze_(0).to("cuda")

            # print(len(StyledImages_F1))
            print(StyledImages_F1.shape)

            StyledImages_F1 = torch.stack(StyledImages_F1)
            StyledImages_F2 = torch.stack(StyledImages_F2)

            

            '''
            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            '''

            Of_model = raft_large(pretrained=True, progress=False).to("cuda")
            Of_model = Of_model.eval()

            Of_Styled = Of_model(StyledImages_F1, StyledImages_F2)
            ### DIMITRI'S COMMENT: Use the original images
            Of_Original = Of_model(images_HED_F1, images_HED_F2)

            ### DIMITRI'S COMMENT: Visualize how the optical flow looks like every n iterations

            print(Of_Original.shape)
            print(Of_Styled.shape)
            
            target = batch[2].float()
            target.requires_grad_()
            model_pred = batch[1].float()
            model_pred.requires_grad_()

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

        # Checks if the accelerator has performed an optimization step behind the scenes
        save_steps = 500
        no_safe_serialization = False
        checkpointing_steps = 500
        checkpoints_total_limit = 50
        validation_prompt = "A man in van gogh style"
        
        if accelerator.sync_gradients:
            images = []
            progress_bar.update(1)
            global_step += 1
            if global_step % save_steps == 0:
                weight_name = (
                    f"learned_embeds-steps-{global_step}.bin"
                    if no_safe_serialization
                    else f"learned_embeds-steps-{global_step}.safetensors"
                )
                save_path = os.path.join(output_dir, weight_name)
                save_progress(
                    text_encoder,
                    placeholder_token_ids,
                    accelerator,
                    placeholder_token,
                    save_path,
                    safe_serialization=not no_safe_serialization,
                )

            if accelerator.is_main_process:
                if global_step % checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if checkpoints_total_limit is not None:
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                ### TODO: Maybe incorporate the log validation function
                ### Also, make better sense to create a dictionary for args first 
                
                # if validation_prompt is not None and global_step % validation_steps == 0:
                #     images = log_validation(
                #         text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
                #     )
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_train_steps:
            break
    ### TODO: At the end of each epoch, update the training loop such that the newly trained model is used

# Create the pipeline using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_full_model = True
    
    if save_full_model:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
	    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer
        )
        pipeline.save_pretrained(output_dir)
    # Save the newly trained embeddings
    weight_name = "learned_embeds.bin" if no_safe_serialization else "learned_embeds.safetensors"
    save_path = os.path.join(output_dir, weight_name)
    save_progress(
        text_encoder,
        placeholder_token_ids,
        accelerator,
        placeholder_token,
        save_path,
        safe_serialization=not no_safe_serialization,
    )

accelerator.end_training()


### TODO: Make variables into a config dictionary
print("Done!")
