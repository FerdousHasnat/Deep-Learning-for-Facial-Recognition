from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from evaluation_model import SimpleCNN, Variant1, Variant2


model_original = SimpleCNN()
print(model_original)

model_variant2 = Variant2()
print(model_variant2)

# Function to evaluate the model on validation set
def evaluate_model(val_loader, model):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    val_accuracy = correct / total
    val_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    val_recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    val_f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    metrics = {
        'val_accuracy': val_accuracy,
        'val_precision_macro': val_precision_macro,
        'val_recall_macro': val_recall_macro,
        'val_f1_macro': val_f1_macro,
        'val_precision_micro': val_precision_micro,
        'val_recall_micro': val_recall_micro,
        'val_f1_micro': val_f1_micro
    }
    return metrics

#transforms
transform = transforms.Compose([
    transforms.Grayscale(),  # If the images are not already in grayscale
    transforms.Resize((48, 48)),  # Resize if the images are not already 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize with mean and std dev
])

# Function to perform K-Fold cross-validation using the saved model
def cross_validate_model(model_path, dataset, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (_, val_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{n_splits}')
        val_subset = Subset(dataset, val_ids)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = Variant2()  # Create the model instance
        model.load_state_dict(torch.load(model_path))  # Load the saved model weights
        model.eval()  # Set the model to evaluation mode

        fold_metrics = evaluate_model(val_loader, model)  # Evaluate the model
        fold_results.append(fold_metrics)
        print(f'Fold {fold+1} metrics: {fold_metrics}')

    avg_metrics = {key: np.mean([fold_result[key] for fold_result in fold_results]) for key in fold_results[0].keys()}
    print("Average metrics across all folds:")
    print(avg_metrics)



# train and test datasets
train_dataset = ImageFolder(root= os.getcwd() + "\\datasets\\final_clean\\train", transform=transform)
test_dataset = ImageFolder(root= os.getcwd() + "\\datasets\\final_clean\\test", transform=transform)

# Combine the dataset
combined_dataset = ConcatDataset([train_dataset, test_dataset])


# Path to saved model 
#model_path = os.getcwd() + "\\saved models\\old models for reference\\part2\\best_model_variant2.pth"
model_path = os.getcwd() + "\\saved models\\best_model_variant2_retrained2.pth"

# Perform K-Fold cross-validation
cross_validate_model(model_path,combined_dataset, n_splits=10)