import os

root_folder = os.getcwd() + '/dataset/unclean/'

output_file = 'dark_image_list.txt'

def list_image_files(folder):
    image_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_files.append(os.path.relpath(os.path.join(root, file), root_folder))
    return image_files

image_files = list_image_files(root_folder)

with open(output_file, 'w') as f:
    for file in image_files:
        f.write('\\' + file + '\n')

print(f"List of image file names written to {output_file} successfully!")