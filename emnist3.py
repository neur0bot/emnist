import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# ===================
# 轻量版 CNN 模型
# ===================
class SmallDigitNet(nn.Module):
    def __init__(self):
        super(SmallDigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)   # 16 通道
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # 32 通道
        self.fc1 = nn.Linear(4608, 128)       # 小型全连接
        self.fc2 = nn.Linear(128, 10)         # 仅 10 类 (digits)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================
# 训练函数
# ===================
def train(model, device, train_loader, optimizer, epoch, train_acc_list):
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    acc = 100. * correct / total
    train_acc_list.append(acc)
    print(f"Train Epoch {epoch}: Accuracy {acc:.2f}%")

# ===================
# 测试函数
# ===================
def test(model, device, test_loader, epoch, test_acc_list):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    test_acc_list.append(acc)
    print(f"Test Epoch {epoch}: Accuracy {acc:.2f}%")
    return acc

# ===================
# 主函数
# ===================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据加载 (EMNIST digits, 10 类)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('./data', split='digits', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('./data', split='digits', train=False, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    model = SmallDigitNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_list, test_acc_list = [], []
    num_epochs = 5   # 少跑几轮就能 >97%

    for epoch in range(1, num_epochs+1):
        train(model, device, train_loader, optimizer, epoch, train_acc_list)
        test(model, device, test_loader, epoch, test_acc_list)

    # 绘制曲线
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_epochs+1), train_acc_list, label="Train Accuracy")
    plt.plot(range(1, num_epochs+1), test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Lightweight EMNIST Digits Accuracy Curve")
    plt.legend()
    plt.grid(True)

    out_png = os.path.join(os.path.dirname(__file__), "test_accuracy3.png")
    plt.savefig(out_png)
    print(f"最终测试准确率：{test_acc_list[-1]:.2f}%，已保存图像到 {out_png}")
    
    plt.show()

if __name__ == "__main__":
    main()
