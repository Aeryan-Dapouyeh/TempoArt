from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import cv2
from PIL import Image
import imageio



def pil_images_to_mp4(image_list, output_path, fps=30):
    # Initialize an ImageIO writer to create the video
    writer = imageio.get_writer(output_path, fps=fps)

    for pil_image in image_list:
        # Convert PIL image to NumPy array
        numpy_image = imageio.core.asarray(pil_image)

        # Append the image to the video writer
        writer.append_data(numpy_image)

    # Close the video writer
    writer.close()


controlnet = ControlNetModel.from_pretrained("CiaraRowles/TemporalNet")

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

# Open the image file.
style_img = Image.open('starry_night_full.jpg')

# Convert the image to a PIL image.
style_img = style_img.convert('RGB')

GeneratedImages = []

# Open the video file.
vidcap = cv2.VideoCapture('DeathSceneTitanic.mp4')

# Read the first frame.
success, image = vidcap.read()
print(success)

# Loop through the frames and extract each one.
count = 0
while success:


    # Convert the frame to a PIL image.
    image_pil = Image.fromarray(image)
    

    # Save the frame as a JPEG file.
    # image_pil.save(f'frame{count}.jpg')
    image = pipeline(image = image_pil, prompt="A man laughing", guidance_scale=20).images[0]
    GeneratedImages.append(image)

    # Read the next frame.
    success, image = vidcap.read()

    # Increment the frame count.
    count += 1

# Example usage:
output_path = "output.mp4"
fps = 30

# Call the function to create the MP4 video
pil_images_to_mp4(GeneratedImages, output_path, fps)
