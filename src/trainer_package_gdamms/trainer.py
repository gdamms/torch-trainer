import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
# from tensorboard.program import TensorBoard
# from tensorboard.default import PLUGIN_LOADERS, get_assets_zip_provider

from typing import Callable, Iterable

import datetime
from .trainer_progress import TrainProgress


class Trainer:
    """A class which trains models."""

    def __init__(self, log_dir: str = f'runs.log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'):
        """Initialize the trainer."""
        self.progress: TrainProgress | None = None
        self.writer = SummaryWriter(log_dir=log_dir)
        # self.tensorboard = TensorBoard(PLUGIN_LOADERS, get_assets_zip_provider())
        # self.tensorboard.configure(argv=['--logdir', log_dir])
        # url = self.tensorboard.launch()
        # print(f"Tensorflow started on {url}")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader[torch.Tensor],
        epochs: int,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_loader: DataLoader[torch.Tensor] | None = None,
        test_loader: DataLoader[torch.Tensor] | None = None,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {},
        epoch_callbacks: Iterable[Callable[[int, int, nn.Module, 'Trainer'], None]] = [],
    ):
        """Train the model for the given number of epochs.

        Args:
            model (nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            epochs (int): The number of epochs to train the model for.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            val_loader (torch.utils.data.DataLoader, optional): The validation dataset. Defaults to None.
            test_loader (torch.utils.data.DataLoader, optional): The test dataset. Defaults to None.
            metrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The metrics to use. Defaults to [].
            epoch_callbacks (List[Callable[[int, nn.Module], None]], optional): The callbacks to call at the end of each epoch. Defaults to [].
        """
        self.date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.progress = TrainProgress(
            nb_epochs=epochs,
            train_size=len(train_loader),
            val_size=len(val_loader) if val_loader else 0,
            test_size=len(test_loader) if test_loader else 0,
        )
        with self.progress:
            for epoch_i in range(epochs):
                self.train_epoch(
                    epoch_i,
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    metrics,
                )
                if val_loader:
                    self.validate(
                        epoch_i,
                        model,
                        val_loader,
                        {'loss': criterion} | metrics,
                    )
                for callback in epoch_callbacks:
                    callback(epoch_i, epochs, model, self)
            if test_loader:
                self.test(
                    model,
                    test_loader,
                    {'loss': criterion} | metrics,
                )
        self.writer.close()

    def train_epoch(
        self,
        epoch_i: int,
        model: nn.Module,
        train_loader: DataLoader[torch.Tensor],
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Train the model for one epoch.

        Args:
            epoch_i (int): The current epoch.
            model (nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            metrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        assert self.progress is not None  # Ensure the progress bar is initialized.

        model.train()
        for batch in train_loader:
            # Seprarate the inputs and labels.
            inputs = batch[:-1]
            labels = batch[-1]

            # Train the model.
            optimizer.zero_grad()
            output = model(*inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Update the progress bar.
            values = {'train_loss': loss.item()} | {f'train_{name}': metric(output, labels).item() for name, metric in metrics.items()}
            self.progress.step()
            self.progress.new_train_values(values)

        self.writer.add_scalars('Loss/train', {self.date_time: loss.item()}, epoch_i)

    def validate(
        self,
        epoch_i: int,
        model: nn.Module,
        val_loader: DataLoader[torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Validate the model on the given validation dataset.

        Args:
            epoch_i (int): The current epoch.
            model (nn.Module): The model to validate.
            val_loader (torch.utils.data.DataLoader): The validation dataset.
            mectrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        assert self.progress is not None  # Ensure the progress bar is initialized.

        model.eval()
        with torch.no_grad():
            metrics_sum = {f'valid_{name}': 0.0 for name in metrics}
            for b_i, batch in enumerate(val_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'valid_{name}': metric(output, labels) for name, metric in metrics.items()}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_val_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })

            self.writer.add_scalars('Metrics/valid', metrics_sum, epoch_i)

    def test(
        self,
        model: nn.Module,
        test_loader: DataLoader[torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Test the model on the given test dataset.

        Args:
            model (nn.Module): The model to test.
            test_loader (torch.utils.data.DataLoader): The test dataset.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            mectrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        assert self.progress is not None  # Ensure the progress bar is initialized.

        model.eval()
        with torch.no_grad():
            metrics_sum = {f'test_{name}': 0.0 for name in metrics}
            for b_i, batch in enumerate(test_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'test_{name}': metric(output, labels) for name, metric in metrics.items()}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_test_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })
