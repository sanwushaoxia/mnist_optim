"""模型定义：MnistCNN（小 CNN）与 ResNet18MNIST（大模型普适性验证）。"""
import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """3 组 Conv-BN-ReLU + MaxPool，2 个 FC 层，约 200 万参数。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18MNIST(nn.Module):
    """torchvision ResNet-18 的 MNIST 适配版，约 1100 万参数。

    conv1 改为 3×3/stride1（原 7×7/stride2 对 28×28 过激）并去掉 maxpool，
    保留 1 通道输入（数据管线无需改动）。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        from torchvision.models.resnet import ResNet, BasicBlock
        net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()  # 28×28 分辨率不再下采样
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """模型工厂：'mnist_cnn' / 'resnet18'。"""
    if name == "mnist_cnn":
        return MnistCNN(num_classes)
    if name == "resnet18":
        return ResNet18MNIST(num_classes)
    raise ValueError(f"未知模型: {name}")


if __name__ == "__main__":
    for name in ("mnist_cnn", "resnet18"):
        m = build_model(name)
        n_params = sum(p.numel() for p in m.parameters())
        out = m(torch.randn(2, 1, 28, 28))
        print(f"{name}: {n_params:,} 参数, 输出 {tuple(out.shape)}, "
              f"fp32 理论 {n_params * 4 / 1024 / 1024:.2f} MB")
