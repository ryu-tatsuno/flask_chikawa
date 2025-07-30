import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from pathlib import Path
import json

# 出力ディレクトリの作成
output_path = Path("apps/detector")
output_path.mkdir(parents=True, exist_ok=True)

# データの前処理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# データ読み込み
dataset = datasets.ImageFolder("dataset", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# クラス名とインデックスの対応を保存（後で推論用に使う）
with open(output_path / "class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump(dataset.class_to_idx, f)

# モデル構築
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # クラス数に応じて調整
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 学習ループ
for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}%")

# モデルの保存（state_dict 形式で保存）
torch.save(model.state_dict(), output_path / "model.pt")

print("学習完了 & モデル保存しました。")
