import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from train_loader import get_dataloaders
from cnn_cifar10 import SimpleCNN

#Denormalisierungsfunktion für die bessere Darstellung falsch klassifizierter Bilder
def denormalize(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img.device).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010], device=img.device).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)

#Device erstellen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DataLoader --> nur für die Testdaten
_, _, test_loader = get_dataloaders(batch_size=64)

#Modell laden
model = SimpleCNN().to(device)
state_dict = torch.load("cnn_cifar10_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

criterion = nn.CrossEntropyLoss()

#Test evalution
test_loss = 0.0
correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
test_loss /= len(test_loader)
test_accuracy = 100*correct/total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")


#Konfusionsmatrix
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Konfusionsmatrix – CIFAR-10")
plt.tight_layout()
plt.show()

#Anzeigen falsch klassifizierter Bilder
examples_shown = 0
max_examples = 10

plt.figure(figsize=(15, 6))

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(images.size(0)):
            if predicted[i] != labels[i]:
                img = denormalize(images[i])
                img = img.cpu().permute(1, 2, 0)

                true_label = class_names[labels[i].item()]
                pred_label = class_names[predicted[i].item()]

                plt.subplot(2, 5, examples_shown + 1)
                plt.imshow(img)
                plt.title(f"True: {true_label}\nPred: {pred_label}")
                plt.axis("off")

                examples_shown += 1
                if examples_shown >= max_examples:
                    break
        if examples_shown >= max_examples:
            break

plt.suptitle("Falsch klassifizierte Beispiele – CIFAR-10")
plt.tight_layout()
plt.show()
