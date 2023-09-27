from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

# controlnet = ControlNetModel.from_pretrained("CiaraRowles/TemporalNet2")
controlnet = ControlNetModel.from_pretrained("CiaraRowles/TemporalNet")

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

pipeline.to("cuda")

# Open the image file.
img = Image.open('ControlNet_diff/starry_night_full.jpg')

# Convert the image to a PIL image.
img = img.convert('RGB')

image = pipeline(image = img, prompt="An image of a squirrel")