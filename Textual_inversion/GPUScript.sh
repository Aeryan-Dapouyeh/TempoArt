#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua40
### -- set the job Name -- 
#BSUB -J TemporalNet_diffusers
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
###BSUB -R "span[hosts=1]"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
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
cd /zhome/70/6/155860/Bachlorproject/TextualInversion
## Write the command to your program here like the following example: 
# python Train_textualInversion_HPC.py
accelerate launch Train_TI_OriginalScript.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --train_data_dir="/work3/s204158/TextualInv_Train/Dataset/train" --learnable_property="style" --placeholder_token="<Temporally_consistent>" --initializer_token="Consistent" --resolution=512 --train_batch_size=2 --gradient_accumulation_steps=4 --max_train_steps=3000 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="/work3/s204158/TextualInv_Train/output"

