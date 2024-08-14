# Deep-Learning-for-Facial-Recognition

Ferdous Hasnat
Aman Nihaal Nuckchady
Lucas Kim


# FER-2013 Dataset
Source: https://www.kaggle.com/datasets/msambare/fer2013/data

Author: Manas Sambare

Description:
The data consists of 48x48 pixel grayscale images of faces. The faces have been
automatically registered so that the face is more or less centered and occupies about the same
amount of space in each image. The task is to categorize each face based on the emotion
shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear,
3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the
public test set consists of 3,589 examples.

Licensing Type: Database: Open Database, Contents: Database Contents (DbCL)


# Emotional Faces Dataset Visualization
This repository contains Python scripts and instructions for visualizing a dataset of images categorized by human emotional expressions. We aim to understand the distribution of classes and pixel intensity variations within the dataset to inform any necessary preprocessing before training a machine learning model.

## Contents
Visualization_01.py: The main Python script that contains functions to analyze and visualize the dataset. This includes the distribution of classes, sample images from each class, and the average pixel intensity distributions.

## Running the Code
To execute the Visualization_01.py script for data visualization, follow these steps:
1. Ensure you have matplotlib and PIL (Pillow) libraries installed in your Python environment
2. Update the train_path and test_path variables in the script to point to the correct location of your dataset on your local machine.
3. Run the script from your terminal or command prompt


## The script will output the following visualizations:
1. A bar graph showing the class distribution of the images.
2. A 5x5 grid of randomly chosen sample images from each class.
3. Histograms displaying the average pixel intensity distribution for each class.

## Understanding the Visualizations

Class Distribution: The bar graph will help you quickly see if the dataset is balanced or if any classes are over or underrepresented.

Sample Images: The 5x5 grid for each class provides a snapshot of the types of images that the model will learn from, and can help in spotting any mislabeled images.

Pixel Intensity Distribution: The histograms for each class show the variations in lighting conditions among the images, which is crucial for understanding the diversity in the dataset.


# Data Cleaning

## Contents
*brightness.py*: The Python script that increases the brightness of all images whose paths are listed in the dark_image_list.txt file.

*list_to_textfile.py*: The Python script that was used to create the dark_image_list.txt file.

*random_transforms.py*: The unused Python script that randomly applies image transforms such as slight rotations and mirroring. 

*dark_image_list.txt*: The file containing a list of paths to the manually sorted darker images that need light augmentation.

## Running the Code
The clean dataset is already provided, so running this code is not necessary. To execute the brightness.py script to apply light augmentation to the unclean dataset, follow these steps:
1. Ensure you have the PIL Python library installed in your Python environment
2. Ensure the script, the unclean dataset folder, and the dark_image_list.txt file are all in the same directory.
3. Run the script from your terminal or command prompt

## Result
Every image whose path is listed in the .txt file will have had its brightness increased by a factor of 1.5.


# Convolutional Neural Network (CNN) Models
This repository contains the Python script cnn_model.py for training Convolutional Neural Networks (CNNs) on image classification tasks.

## Models
Three CNN architectures are included:

SimpleCNN: The base model.
Variant1: An extended version of the base model with an additional convolutional layer.
Variant2: A variation of the base model with different kernel sizes.

## Getting Started
To run the training process, ensure you have the following prerequisites installed:

Python 3.6 or higher,
PyTorch,
torchvision,
PIL (Pillow),
NumPy

## Steps to Run the Training
1. Clone the repository
2. Execute the training script
3. The script will automatically train three models using the specified training and validation datasets

## Model Checkpoints
During training, the best model checkpoints are saved to the file system as best_model_original.pth, best_model_variant1.pth, and best_model_variant2.pth. These files will be used for the evaluation and application of the models.



# Model Evaluation and Application
This repository contains the Python script evaluation_model.py for evaluating pre-trained Convolutional Neural Networks (CNNs) and applying them to classify new images.

## Prerequisites
Before running the evaluation and application, ensure the training process has been completed using cnn_model.py and the model checkpoints are saved.

## Steps to Run the Evaluation
1. Ensure the pre-trained model checkpoints are in the root directory of the project:
best_model_original.pth
best_model_variant1.pth
best_model_variant2.pth
2. The test dataset should be located under the path datasets/final_clean/test/, with subdirectories for each class label containing the respective images
3. Run the evaluation script

This will print out the accuracy of each model on the test dataset and predict the class of a specified image

## Application to Classify New Images
To apply a trained model to new images, modify the image_path variable in the script evaluation_model.py to point to your new image. Then, rerun the script



# Model Evaluation Metrics

This repository contains the Python script eval_metrics.py for evaluating the performance of Convolutional Neural Networks (CNNs) on an image classification task.

## Overview

The script provides functionality to evaluate three distinct CNN architectures:

Original: The baseline CNN model.
Variant1: An augmented version of the baseline with an extra convolutional layer.
Variant2: A variation of the baseline with altered kernel sizes.

## Evaluation Metrics
The evaluation covers the following metrics for each model:

Macro Precision, Recall, and F1 Score

Micro Precision, Recall, and F1 Score

Overall Accuracy

Confusion Matrix

These metrics are crucial for understanding each model's performance and for comparing the effectiveness of the architectural variations

## Prerequisites
Before running the evaluation script, make sure the following prerequisites are installed:

Python 3.6 or higher,
PyTorch,
torchvision,
scikit-learn,
pandas,
PIL (Pillow)

## Running the Evaluation
1. Ensure that you have completed training your models and have saved the best-performing checkpoints for the Original, Variant1, and Variant2 models.
2. Adjust the dataset_path variable in the script to point to the directory containing your test dataset
3. Modify the model_*_path variables to point to the respective saved model checkpoints
4. Run the script

The script will load each model, perform evaluations on the test dataset, and output the computed metrics and confusion matrices

## Output
The results are displayed in the console, and a pandas DataFrame is printed for a tabular view of the metrics

The script will also generate results in the form of CSV files made up of evaluation metrics of each model. These results can be found in "/model evaluation/results/".

Previous results from our own testing are all saved in "/model evaluation/results/model evaluation metrics/".

# K-fold Cross-Validation
This section of the repository contains code for performing K-fold cross-validation on a convolutional neural network model trained on the FER-2013 dataset for facial emotion recognition.

K-fold cross-validation is a statistical method used to estimate the skill of machine learning models. It is particularly useful in scenarios where the goal is to predict the outcome for an unseen data point. This method reduces the variance associated with a single trial of train/test split by dividing the data into K folds and then iteratively training and validating the model K times, each time with a different fold held out for validation.

The script for K-fold cross-validation can be found in "/model evaluation/K-fold Cross-validation.py".

## Model Architecture
We provide two CNN architectures:

SimpleCNN: The baseline convolutional neural network.
Variant2: An improved version with modified kernel sizes and additional layers.
Both models are defined in the models.py file and can be initialized and used for training and evaluation

## Evaluation Function
The evaluate_model function takes a data loader for the validation set and a model as input and returns various performance metrics such as accuracy, precision, recall, and F1 score for both macro and micro averages.

## Cross-validation Function
The cross_validate_model function automates the K-fold cross-validation process by:

Splitting the dataset into training and validation subsets for each fold.
Evaluating the model on each fold using the evaluate_model function.
Collecting and averaging the performance metrics across all folds to provide a robust estimate of the model's performance.

## Running Cross-validation
To run K-fold cross-validation:

1. Ensure the transforms module is properly defined to preprocess the images as required by the model.
2. Prepare the datasets using the ImageFolder class and combine them into a single dataset if necessary.
3. Call the cross_validate_model function with the appropriate model weights path and dataset. The path to the pre-trained weights and the dataset directory must be correctly specified.

# Bias Analysis
Bias analysis was conducted to determine if our model has any noticeable biases across different groups in age and gender.

The script for bias analysis can be found in "/model evaluation/bias_analysis.py".

A custom dataset was also created to test our models for bias, with separate folders for age and gender, and separate folders for each subgroup within them.

This dataset can be found in the main directory of the repository under "/dataset-bias-analysis/".

## Prerequisites
Before running the script, make sure the following prerequisites are installed:

Python 3.6 or higher,
PyTorch,
torchvision,
scikit-learn,
pandas

## Output
Once run, the script will print the confusion matrix for each class for each model as well as display a visualization (more clean version) of each confusion matrix, and a pandas DataFrame is printed for a tabular view of the metrics.

The script will also generate results in the form of CSV files made up of evaluation metrics of the model on each group we tested for bias. These results can be found in "/model evaluation/results/".

Previous results from our own testing are all saved in "/model evaluation/results/bias analysis metrics/".
