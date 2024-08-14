import os
from PIL import ImageEnhance
from PIL import Image

# set root directory of dataset
root_dir = os.getcwd() + '\\dataset'

# set path to text file containing list of sorted dark images
text_file_path = os.getcwd() + '\\dark_image_list.txt'

# set brightness factor
brightness_factor = 1.5

# read the text file and create a list of image paths
with open(text_file_path, 'r') as file:
    lines = file.readlines()

# iterate through the list of image paths and apply brightness increase
for line in lines:
    line = line.strip()
    if line:
        image_path = root_dir + line
        
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            
            enhancer = ImageEnhance.Brightness(image)
            adjusted_image = enhancer.enhance(brightness_factor)

            adjusted_image.save(image_path)

print("Brightness increase applied to listed images!")
