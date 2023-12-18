from PIL import Image
import os

def extract_frame_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def resize_img(input_dir):
    image_dir = os.path.join(".", f"{input_dir}")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))], key = extract_frame_number)

    image_files[0]

    images_PIL = [Image.open(os.path.abspath(os.path.join(image_dir, image)), mode='r') for image in image_files]

    images_resize = [image.resize((1920, 1080)) for image in images_PIL if image.size[0] > 1920]

    images_jpg = [img.convert("RGB") for img in images_resize]

    for idx, rgb_im in enumerate(images_jpg):
        rgb_im.save(os.path.join(input_dir, f"frame{idx}.jpg"))