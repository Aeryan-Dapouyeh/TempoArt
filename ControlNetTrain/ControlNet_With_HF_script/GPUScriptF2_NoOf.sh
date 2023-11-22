#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J TemporalNet_diffusers
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
###BSUB -R "span[hosts=1]"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 12GB of system-memory
#BSUB -R "rusage[mem=12GB]"
###BSUB -R "select[gpu80gb]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
### BSUB -M 3GB

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s204158@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
nvidia-smi
module load cuda/11.6
cd /zhome/70/6/155860
source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate DL_CompVision
cd /zhome/70/6/155860/Bachlorproject/ControlNet_Train_OriginalScript
## Write the command to your program here like the following example: 
# python Train_textualInversion_HPC.py
accelerate launch train_controlnet_modified_F2_NoOf.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --train_data_dir="/work3/s204158/VideoDataSet/ProcessedData" --resolution=512 --learning_rate=1e-5 --validation_image "/zhome/70/6/155860/Bachlorproject/ControlNet_Train_OriginalScript/ValidationImages/1/F1.png" --validation_prompt "A man dancing in van gogh style, masterpiece" --train_batch_size=4 --num_train_epochs=3 --tracker_project_name="controlnet" --enable_xformers_memory_efficient_attention --checkpointing_steps=10000 --validation_steps=500 --max_train_steps=100000
