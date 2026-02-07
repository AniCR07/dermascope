import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\RAAM\Downloads\DermaDetectAI-main\sd-198"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
NUM_CLASSES = 10   # change to 10 or 23 accordingly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Class order:", train_dataset.classes)

# ---------------- MODEL ----------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "skin_disease_model.pth")
print("Model saved as skin_disease_model.pth")
