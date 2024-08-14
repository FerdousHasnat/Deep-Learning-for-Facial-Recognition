# %%
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

train_path = os.getcwd() + "\\datasets\\final_clean\\train"
test_path = os.getcwd() + "\\datasets\\final_clean\\test"

# This function will go through the dataset folder and return information for visualization
def analyze_dataset(path):
    class_distribution = {}
    sample_images_by_class = {}
    pixel_intensity_distributions = {}

    # Iterate over each class folder
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            class_distribution[class_name] = len(images)  # Count images per class
            selected_images = random.sample(images, 25)  # Randomly select 25 images
            sample_images_by_class[class_name] = [os.path.join(class_dir, img) for img in selected_images]

            # Read the images and calculate pixel intensity distribution
            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                # Convert to grayscale
                img = Image.open(img_path).convert('L')  
                img_array = np.array(img)
                pixel_intensity_distributions.setdefault(class_name, []).append(img_array.flatten())

    return class_distribution, sample_images_by_class, pixel_intensity_distributions

def visualize_class_distribution(class_distribution):
    
    plt.figure(figsize=(10, 5))
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    plt.bar(classes, counts, color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)  # Rotate class labels to fit them nicely
    plt.show()

def visualize_sample_images(sample_images_by_class):
    
    for class_name, image_paths in sample_images_by_class.items():
        fig, axes = plt.subplots(5, 5, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
        fig.suptitle(f'Sample Images for Class: {class_name}', fontsize=16)
        for i, ax in enumerate(axes.flat):
            img = Image.open(image_paths[i]).convert('L')  # Convert image to grayscale
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()


# %%
def visualize_average_pixel_intensity_distribution(pixel_intensity_distributions):
    
    bins = 256
    
    # Iterate over each class in the pixel intensity distribution
    for class_name, intensity_lists in pixel_intensity_distributions.items():
        # Concatenate all intensity lists into a single array
        all_intensities = np.concatenate(intensity_lists, axis=0)
        plt.figure(figsize=(10, 5))
        
        # histogram for all intensities
        plt.hist(all_intensities, bins=bins, alpha=0.7, label=class_name, density=True)
        
        # title and labels
        plt.title(f'Average Pixel Intensity Distribution for Class: {class_name}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.legend()  
        plt.show()

class_distribution, sample_images_by_class, pixel_intensity_distributions = analyze_dataset(train_path)

# The visualization functions
visualize_class_distribution(class_distribution)
visualize_sample_images(sample_images_by_class)
visualize_average_pixel_intensity_distribution(pixel_intensity_distributions)
    


# %%



