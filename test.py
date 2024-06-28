import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.trainer_package_gdamms.trainer import Trainer


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


def main():
    trainer = Trainer()
    model = Model()
    dataset = datasets.MNIST(
        root='data.log',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.flatten(),
        ]),
        target_transform=transforms.Compose([
            lambda x: torch.tensor(x),
            lambda x: F.one_hot(x, 10),
            lambda x: x.float(),
        ]),
    )
    dataset_val = datasets.MNIST(
        root='data.log',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.flatten(),
        ]),
        target_transform=transforms.Compose([
            lambda x: torch.tensor(x),
            lambda x: F.one_hot(x, 10),
            lambda x: x.float(),
        ]),
    )
    data_loader: DataLoader[torch.Tensor] = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    data_loader_val: DataLoader[torch.Tensor] = DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=4)
    trainer.train(
        model,
        data_loader,
        epochs=5,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=torch.nn.CrossEntropyLoss(),
        metrics={'crosse': torch.nn.functional.cross_entropy},
        val_loader=data_loader_val,
        test_loader=data_loader_val,
    )


if __name__ == '__main__':
    main()
