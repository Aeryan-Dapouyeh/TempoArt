import argparse
import os
import subprocess
from torchvision.io.video import read_video, write_video

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataDirectory", type=str)
parser.add_argument("-o", "--outputDirectory", type=str)
parser.add_argument("-f", "--pathToOFDirectory", type=str)
parser.add_argument("-m", "--modelName", type=str)

args = parser.parse_args()

dataDirectory = args.dataDirectory
outputDirectory = args.outputDirectory
pathToOFDirectory = args.pathToOFDirectory
modelName = args.modelName

cwd = os.getcwd()

#modelPath = "/pretrained/{}".format(modelName)
#modelPath = os.path.join(pathToOFDirectory, modelPath)
modelPath = "C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/TempoArt/OpticalFlow/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"

pathToOFDirectory = os.path.join(cwd, pathToOFDirectory)

pathToscript = "C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/TempoArt/OpticalFlow/unimatch/main_flow.py"
dataDirectory = "C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/TempoArt/Input_baseline_images/HED_vanGogh/resized"
outputDirectory = "C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/TempoArt/OpticalFlow/Generate_OF/output"


args = [
    'python', pathToscript,
    '--inference_video', 'None',
    '--resume', modelPath,
    '--output_path', outputDirectory,
    '--padding_factor', '32',
    '--upsample_factor', '4',
    '--num_scales', '2',
    '--attn_splits_list', '2', '8',
    '--corr_radius_list', '-1', '4',
    '--prop_radius_list', '-1', '1',
    '--reg_refine',
    '--num_reg_refine', '6',
    '--save_video'
]

for filename in os.listdir(dataDirectory):
    if filename.endswith('.mp4'):
        print("An Mp4 file!")
        videoPath = f"{dataDirectory}/{filename}"
        args[3] = videoPath
        subprocess.run(args)


# Example of a terminal command
# python OpticalFlow\Generate_OF\Generate_opticalFlow.py -d Input_baseline_images -o output -f OpticalFlow/unimatch -m OpticalFlow\unimatch\pretrained\gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
