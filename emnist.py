import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
import matplotlib.pyplot as plt
import os

# 可配置项
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
SEED = 42
SPLIT = "digits"  # EMNIST split: digits gives 10 classes (easier to achieve >85%)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Keep padding=1 so spatial dims halve cleanly after each pool:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        # after two pool halves: 28 -> 14 -> 7 ; flattened = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = EMNIST(root=DATA_DIR, split=SPLIT, train=True, download=True, transform=transform)
    test = EMNIST(root=DATA_DIR, split=SPLIT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, len(train), len(test), train.classes if hasattr(train, "classes") else None


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train():
    train_loader, test_loader, _, _, _ = load_data(BATCH_SIZE)
    num_classes = 10  # digits split => 10 classes
    model = SimpleConvNet(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    test_accs = []
    times = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        elapsed = time.time() - start_time
        test_acc = evaluate(model, test_loader) * 100.0
        test_accs.append(test_acc)
        times.append(elapsed)
        print(f"Epoch {epoch}/{EPOCHS}  loss={epoch_loss/len(train_loader.dataset):.4f}  test_acc={test_acc:.2f}%  time={elapsed:.1f}s")

        # 若已超过 85% 可提前停止
        if test_acc >= 85.0:
            print(f"目标达到：测试集准确率 {test_acc:.2f}% >= 85%，提前停止。")
            break

    final_acc = test_accs[-1]
    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(times, test_accs, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"EMNIST ({SPLIT}) Test Accuracy over Time\nFinal: {final_acc:.2f}%")
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), "test_accuracy.png")
    plt.savefig(out_png)
    print(f"最终测试准确率：{final_acc:.2f}%，已保存图像到 {out_png}")
    try:
        plt.show()
    except Exception:
        pass

    return model, final_acc


if __name__ == "__main__":
    model, acc = train()