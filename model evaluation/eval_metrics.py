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

#  dataset
dataset_path = os.getcwd() + "/datasets/final_clean/test"

# dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training, validation, and test sets
#_, test_data = train_test_split(dataset, test_size=0.15, random_state=42)

# DataLoader
test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)


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


# pre-trained models location
model_original_path = os.getcwd() + "/saved models/old models for reference/part3/best_model_original_retrained2.pth"
model_variant1_path = os.getcwd() + "/saved models/old models for reference/part3/best_model_variant1_retrained2.pth"
model_variant2_path = os.getcwd() + "/saved models/best_model_variant2_retrained2.pth"

# Load the models
model_original = load_model(model_original_path, SimpleCNN)
model_variant1 = load_model(model_variant1_path, Variant1)
model_variant2 = load_model(model_variant2_path, Variant2)

# Evaluate the models
original_eval = evaluate_model(model_original, test_loader)
variant1_eval = evaluate_model(model_variant1, test_loader)
variant2_eval = evaluate_model(model_variant2, test_loader)

# Display the confusion matrix for each model
print("Confusion Matrix for Original Model:")
print(original_eval[7])
print("Confusion Matrix for Variant 1:")
print(variant1_eval[7])
print("Confusion Matrix for Variant 2:")
print(variant2_eval[7])

# A DataFrame to show the results
results_df = pd.DataFrame({
    'Model': ['Original', 'Variant 1', 'Variant 2'],
    'Macro Precision': [original_eval[1], variant1_eval[1], variant2_eval[1]],
    'Macro Recall': [original_eval[2], variant1_eval[2], variant2_eval[2]],
    'Macro F1': [original_eval[3], variant1_eval[3], variant2_eval[3]],
    'Micro Precision': [original_eval[4], variant1_eval[4], variant2_eval[4]],
    'Micro Recall': [original_eval[5], variant1_eval[5], variant2_eval[5]],
    'Micro F1': [original_eval[6], variant1_eval[6], variant2_eval[6]],
    'Accuracy': [original_eval[0], variant1_eval[0], variant2_eval[0]]
})

print("\nResults:")
print(results_df)
results_df.to_csv(os.getcwd() + '/model evaluation/results/eval_metrics.csv', index=False)
