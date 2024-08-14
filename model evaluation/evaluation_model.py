import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

class SimpleCNN(nn.Module):
    name = "original"

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # The FER-2013 images are grayscale, so in_channels=1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FER-2013 images are 48x48 pixels, so after two pooling layers, the size will be 12x12.
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes to classify
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Variant 1: Add a third convolutional layer
class Variant1(SimpleCNN):
    name = "variant1"

    def __init__(self):
        super(Variant1, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 3x3 kernel
        self.fc1 = nn.Linear(128 * 6 * 6, 128)  # Adjust the input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Add a forward pass through the third convolutional layer
        x = x.view(-1, 128 * 6 * 6)  # Adjust the size
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Variant 2: Change kernel sizes
class Variant2(SimpleCNN):
    name = "variant2"

    def __init__(self):
        super(Variant2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)  # 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)  # 2x2 kernel
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjust the input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Adjust the size
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def evaluate_on_dataset(model, dataset_path, transform):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Model type: ' + model.name)
    print(f'Accuracy on the dataset: {accuracy:.2f}%')


def predict_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = class_labels[predicted.item()]
        print(f'Predicted class: {predicted_label}\n')


# Test class labels
class_labels = {
    0: "engaged",
    1: "happy",
    2: "neutral",
    3: "surprise"
}

if __name__ == '__main__':
    model_original_path = os.getcwd() + "/saved models/best_model_original.pth"
    model_variant1_path = os.getcwd() + "/saved models/best_model_variant1.pth"
    model_variant2_path = os.getcwd() + "/saved models/best_model_variant2.pth"

    dataset_path = os.getcwd() + "/datasets/final_clean/test"
    image_path = os.getcwd() + "/datasets/final_clean/test/engaged/PrivateTest_11123843.jpg"

    # transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # evaluate original model
    model_original = load_model(model_original_path, model=SimpleCNN())
    evaluate_on_dataset(model_original, dataset_path, transform)  # Evaluate on the entire "test" dataset
    predict_image(model_original, image_path, transform)  # Predict a single image

    # evaluate variant 1
    model_variant1 = load_model(model_variant1_path, model=Variant1())
    evaluate_on_dataset(model_variant1, dataset_path, transform)  # Evaluate on the entire "test" dataset
    predict_image(model_variant1, image_path, transform)  # Predict a single image

    # evaluate variant 2
    model_variant2 = load_model(model_variant2_path, model=Variant2())
    evaluate_on_dataset(model_variant2, dataset_path, transform)  # Evaluate on the entire "test" dataset
    predict_image(model_variant2, image_path, transform)  # Predict a single image
