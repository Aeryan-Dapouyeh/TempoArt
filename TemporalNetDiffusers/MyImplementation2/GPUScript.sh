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
#BSUB -W 12:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=8GB]"

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
cd /zhome/70/6/155860/Bachlorproject/MyImplementation2
python main.py -i "Input/BSVid.mp4" -p "Van gogh style painting of a male dancer dancing, detailed" -o "BSVid_Van_gogh.mp4"
### python main.py -i "Input/DYVid.mp4" -p "Van gogh style painting of a man dancing, detailed" -o "DYVid_Van_gogh.mp4"
python main.py -i "Input/LCVid.mp4" -p "Van gogh style painting of a black man laughing, detailed" -o "LCVid_Van_gogh.mp4"
python main.py -i "Input/LMVid.mp4" -p "Van gogh style painting of a man with a headphone, detailed" -o "LMVid_Van_gogh.mp4"
python main.py -i "Input/RSVid.mp4" -p "Van gogh style painting of woman running, detailed" -o "RSVid_Van_gogh.mp4"
python main.py -i "Input/SBVid.mp4" -p "Van gogh style painting of woman running in the beach, detailed" -o "SBVid_Van_gogh.mp4"
python main.py -i "Input/SCVid.mp4" -p "Van gogh style painting of a woman smiling, detailed" -o "SCVid_Van_gogh.mp4"
python main.py -i "Input/SWVid.mp4" -p "Van gogh style painting of a woman, detailed" -o "SWVid_Van_gogh.mp4"
python main.py -i "Input/WLVid.mp4" -p "Van gogh style painting of a woman working with her computer, detailed" -o "WLVid_Van_gogh.mp4"
python main.py -i "Input/WWVid.mp4" -p "Van gogh style painting of a man walking in the desert, detailed" -o "WWVid_Van_gogh.mp4"

