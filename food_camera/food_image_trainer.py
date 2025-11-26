# 음식 이미지 학습 코드 (EfficientNet 기반, food_db와 연동)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Food101

# 학습 설정
BATCH_SIZE = 32
EPOCHS = 10
NUM_WORKERS = 2

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Food-101 데이터셋 다운로드 및 불러오기
full_dataset = Food101(root="./data", split="train", transform=transform, download=True)

# 학습/검증 데이터 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)

# 모델 불러오기 및 출력층 수정
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "food_classifier.pth")
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\\n")

if __name__ == "__main__":
    train()
