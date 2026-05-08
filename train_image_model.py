"""
train_image_model.py
Fine-tunes ResNet18 on Kaggle chest X-ray dataset (NORMAL / PNEUMONIA).
"""

import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

KAGGLE_TRAIN = r"C:\Users\MASI HASAN ALI\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\train"
KAGGLE_TEST  = r"C:\Users\MASI HASAN ALI\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\test"

BASE        = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, "dataset_images")
SAVE_PATH   = os.path.join(BASE, "image_model.pth")

EPOCHS      = 5
BATCH_SIZE  = 16
LR          = 0.0005
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def prepare_dataset():
    print("\nPreparing dataset...")
    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
    for split_dir in [KAGGLE_TRAIN, KAGGLE_TEST]:
        if not os.path.exists(split_dir):
            continue
        for cls in ["NORMAL", "PNEUMONIA"]:
            src = os.path.join(split_dir, cls)
            dst = os.path.join(DATASET_DIR, cls)
            if not os.path.exists(src):
                continue
            for fname in os.listdir(src):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    dst_file = os.path.join(dst, fname)
                    if not os.path.exists(dst_file):
                        shutil.copy2(os.path.join(src, fname), dst_file)
    print(f"  NORMAL:    {len(os.listdir(os.path.join(DATASET_DIR, 'NORMAL')))} images")
    print(f"  PNEUMONIA: {len(os.listdir(os.path.join(DATASET_DIR, 'PNEUMONIA')))} images\n")

if not os.path.exists(DATASET_DIR):
    prepare_dataset()
else:
    print(f"Dataset ready: NORMAL={len(os.listdir(os.path.join(DATASET_DIR,'NORMAL')))}, PNEUMONIA={len(os.listdir(os.path.join(DATASET_DIR,'PNEUMONIA')))}\n")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
classes = full_dataset.classes
print(f"Classes: {classes} | Total: {len(full_dataset)}")

val_size   = max(1, int(0.15 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}\n")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(out, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(imgs), 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train: {correct/total:.2%} | Val: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved (val_acc={val_acc:.2%})")

print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.2%}")
print(f"   Model saved: {SAVE_PATH}")
print(f"   Restart backend to load new model.")