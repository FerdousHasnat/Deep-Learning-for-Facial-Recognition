from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation_model import SimpleCNN, Variant1, Variant2
import os

# transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# dataset paths age
dataset_age_middle_path = os.getcwd() + "/dataset-bias-analysis/age/middle-aged"
dataset_age_senior_path = os.getcwd() + "/dataset-bias-analysis/age/senior"
dataset_age_young_path = os.getcwd() + "/dataset-bias-analysis/age/young"

# dataset paths gender
dataset_gender_male_path = os.getcwd() + "/dataset-bias-analysis/gender/male"
dataset_gender_female_path = os.getcwd() + "/dataset-bias-analysis/gender/female"

# datasets age
dataset_age_middle = datasets.ImageFolder(root=dataset_age_middle_path, transform=transform)
dataset_age_senior = datasets.ImageFolder(root=dataset_age_senior_path, transform=transform)
dataset_age_young = datasets.ImageFolder(root=dataset_age_young_path, transform=transform)

# datasets gender
dataset_gender_male = datasets.ImageFolder(root=dataset_gender_male_path, transform=transform)
dataset_gender_female = datasets.ImageFolder(root=dataset_gender_female_path, transform=transform)

# DataLoaders for age
test_loader_age_middle = DataLoader(dataset=dataset_age_middle, batch_size=64, shuffle=False)
test_loader_age_senior = DataLoader(dataset=dataset_age_senior, batch_size=64, shuffle=False)
test_loader_age_young = DataLoader(dataset=dataset_age_young, batch_size=64, shuffle=False)

# DataLoaders for gender
test_loader_gender_male = DataLoader(dataset=dataset_gender_male, batch_size=64, shuffle=False)
test_loader_gender_female = DataLoader(dataset=dataset_gender_female, batch_size=64, shuffle=False)


def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def evaluate_model(model, test_loader):

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(labels.numpy())
    accuracy = accuracy_score(all_targets, all_predictions)

    macro_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    micro_precision = precision_score(all_targets, all_predictions, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)

    conf_matrix = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()

    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, conf_matrix


# pre-trained model locations
model_original_path = os.getcwd() + "/saved models/old models for reference/part3/best_model_original_retrained2.pth"
model_variant1_path = os.getcwd() + "/saved models/old models for reference/part3/best_model_variant1_retrained2.pth"
model_variant2_path = os.getcwd() + "/saved models/best_model_variant2_retrained2.pth"

# Load the model
model_original = load_model(model_original_path, SimpleCNN)
model_variant1 = load_model(model_variant1_path, Variant1)
model_variant2 = load_model(model_variant2_path, Variant2)


# evaluate models
for model in [model_original, model_variant1, model_variant2]:   
    # Evaluate the model on age
    eval_age_middle = evaluate_model(model, test_loader_age_middle)
    eval_age_senior = evaluate_model(model, test_loader_age_senior)
    eval_age_young = evaluate_model(model, test_loader_age_young)

    # Evaluate the model on gender
    eval_gender_male = evaluate_model(model, test_loader_gender_male)
    eval_gender_female = evaluate_model(model, test_loader_gender_female)

    # Display the confusion matrix for age for the model
    print("Confusion Matrix for age middle for " + model.name + " Model:")
    print(eval_age_middle[7])

    print("Confusion Matrix for age senior for " + model.name + " Model:")
    print(eval_age_senior[7])

    print("Confusion Matrix for age young for " + model.name + " Model:")
    print(eval_age_young[7])

    # Display the confusion matrix for gender for the model
    print("Confusion Matrix for gender male for " + model.name + " Model:")
    print(eval_gender_male[7])

    print("Confusion Matrix for gender female for " + model.name + " Model:")
    print(eval_gender_female[7])


    # A DataFrame to show the results for age
    results_df_age = pd.DataFrame({
        'Age': ['Young', 'Middle-Aged', 'Senior'],
        'Macro Precision': [eval_age_young[1], eval_age_middle[1], eval_age_senior[1]],
        'Macro Recall': [eval_age_young[2], eval_age_middle[2], eval_age_senior[2]],
        'Macro F1': [eval_age_young[3], eval_age_middle[3], eval_age_senior[3]],
        #'Micro Precision': [eval_age_young[4], eval_age_middle[4], eval_age_senior[4]],
        #'Micro Recall': [eval_age_young[5], eval_age_middle[5], eval_age_senior[5]],
        #'Micro F1': [eval_age_young[6], eval_age_middle[6], eval_age_senior[6]],
        'Accuracy': [eval_age_young[0], eval_age_middle[0], eval_age_senior[0]]
    })

    # A DataFrame to show the results for gender
    results_df_gender = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Macro Precision': [eval_gender_male[1], eval_gender_female[1]],
        'Macro Recall': [eval_gender_male[2], eval_gender_female[2]],
        'Macro F1': [eval_gender_male[3], eval_gender_female[3]],
        #'Micro Precision': [eval_gender_male[4], eval_gender_female[4]],
        #'Micro Recall': [eval_gender_male[5], eval_gender_female[5]],
        #'Micro F1': [eval_gender_male[6], eval_gender_female[6]],
        'Accuracy': [eval_gender_male[0], eval_gender_female[0]]
    })

    print("\nResults for age:")
    print(results_df_age)
    results_df_age.to_csv(os.getcwd() + '/results/' + model.name +'_retrained2_bias_metrics_age.csv', index=False)

    print("\nResults for gender:")
    print(results_df_gender)
    results_df_gender.to_csv(os.getcwd() + '/results/' + model.name +'_retrained2_bias_metrics_gender.csv', index=False)
