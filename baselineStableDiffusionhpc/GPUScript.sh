#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J BaseLine
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
### BSUB -M 3GB

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s204134@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
export HF_DATASETS_CACHE="/work3/s204134"
nvidia-smi
module load cuda/11.6
cd /work3/s204134
source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate BestEnv
cd /work3/s204134/baselineStableDiffusionhpc

# Example of a scipt with Generate_baseline.py 
python Generate_baseline.py -i "movieClips/BSVid.mp4" -o "BSVid_Baseline.mp4" -p "Van gogh style painting of a male dancer dancing, detailed" -c "/work3/s204158/HF_cache"

### python main.py
### "Van gogh style painting of a male dancer dancing, detailed"
### "Van gogh style painting of a man dancing, detailed"
### "Van gogh style painting of a black man laughing, detailed"
### "Van gogh style painting of a man with a headphone, detailed"
### "Van gogh style painting of woman running, detailed"
### "Van gogh style painting of woman running in the beach, detailed"
### "Van gogh style painting of a woman smiling, detailed"
### "Van gogh style painting of a woman, detailed"
### "Van gogh style painting of a woman working with her computer, detailed"
### "Van gogh style painting of a man walking in the desert, detailed"





