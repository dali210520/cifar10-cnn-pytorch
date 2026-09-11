import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt

#Damit wir die Daten in cnn_cifar10.py nutzen können
def get_dataloaders(batch_size=64, data_path="./CIFAR-10"):

    #Die Funktion transformiert bzw. normiert die Werte von [0,255] auf [0,1]
    ttransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform
    )

    #Jetzt wird eine Aufteilung in Trainings- und Validierungsdaten vorgenommen
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )

    #Jetzt erstellen wir noch einen Datenloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



#Der Teil ist für die Exploration der Daten
#Die Funktion transformiert bzw. normiert die Werte von [0,255] auf [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    #Durchschnittswerte und Standardabweichung des CIFAR-10 Datensatzes
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

data_path = "./CIFAR-10"  # Ordner, in dem CIFAR-10 gespeichert ist

train_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    download= False,   
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    download= False,
    transform=transform
)

#Klassenbezeichnungen
class_names = train_dataset.classes
print("Klassen:", class_names)


#Erste Exploration der Daten
#Klassenverteilung
labels = [label for _, label in train_dataset]
class_counts = Counter(labels)

#Gibt aus welche Klasse wie viele Daten besitzt
print("\nKlassenverteilung im Trainingsdatensatz:")
for idx, count in class_counts.items():
    print(f"{class_names[idx]}: {count}")
    

#Jetzt wird eine Aufteilung in Trainings- und Validierungsdaten vorgenommen
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size]
)

print("\nDatensatzgrößen:")
print("Training:", len(train_dataset))
print("Validierung:", len(val_dataset))
print("Test:", len(test_dataset))

#Jetzt erstellen wir noch einen Datenloader
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nDataLoader erfolgreich erstellt.")


