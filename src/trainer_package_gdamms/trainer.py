import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.program import TensorBoard
from tensorboard.default import PLUGIN_LOADERS, get_assets_zip_provider


from typing import Callable, Collection, Iterable

import datetime
from .trainer_progress import TrainProgress


class Trainer:
    """A class which trains models."""

    def __init__(self, log_dir: str = 'runs'):
        """Initialize the trainer."""
        self.progress: TrainProgress | None = None
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tensorboard = TensorBoard(PLUGIN_LOADERS, get_assets_zip_provider())
        self.tensorboard.configure(argv=['--logdir', log_dir])
        url = self.tensorboard.launch()
        print(f"Tensorflow started on {url}")

    def train(
        self,
        model: nn.Module,
        train_loader: Collection[torch.Tensor],
        epochs: int,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_loader: Collection[torch.Tensor] | None = None,
        test_loader: Collection[torch.Tensor] | None = None,
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [],
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
            metrics (List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The metrics to use. Defaults to [].
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
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    metrics,
                    epoch_i,
                )
                if val_loader:
                    self.validate(
                        model,
                        val_loader,
                        metrics + [criterion],
                    )
                for callback in epoch_callbacks:
                    callback(epoch_i, epochs, model, self)
            if test_loader:
                self.test(
                    model,
                    test_loader,
                    metrics + [criterion],
                )
        self.writer.close()

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: Collection[torch.Tensor],
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        epoch_i: int,
    ):
        """Train the model for one epoch.

        Args:
            model (nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): The training dataset.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            metrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
            epoch_i (int): The current epoch.
        """
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
            values = {metric.__name__: metric(output, labels)
                      for metric in metrics}
            values[criterion.__name__] = loss.item()
            self.progress.step()
            self.progress.new_train_values(values)

        self.writer.add_scalars('Criterion/train', {self.date_time: loss.item()}, epoch_i)

    def validate(
        self,
        model: nn.Module,
        val_loader: Collection[torch.Tensor],
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Validate the model on the given validation dataset.

        Args:
            model (nn.Module): The model to validate.
            val_loader (torch.utils.data.DataLoader): The validation dataset.
            mectrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        model.eval()
        with torch.no_grad():
            metrics_sum = {f'val_{metric.__name__}': 0 for metric in metrics}
            for b_i, batch in enumerate(val_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'val_{metric.__name__}': metric(output, labels)
                          for metric in metrics}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_val_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })

    def test(
        self,
        model: nn.Module,
        test_loader: Collection[torch.Tensor],
        metrics: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Test the model on the given test dataset.

        Args:
            model (nn.Module): The model to test.
            test_loader (torch.utils.data.DataLoader): The test dataset.
            mectrics (list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): The metrics to use.
        """
        model.eval()
        with torch.no_grad():
            metrics_sum = {f'test_{metric.__name__}': 0 for metric in metrics}
            for b_i, batch in enumerate(test_loader):
                inputs = batch[:-1]
                labels = batch[-1]
                output = model(*inputs)
                values = {f'test_{metric.__name__}': metric(output, labels)
                          for metric in metrics}
                for key, value in values.items():
                    metrics_sum[key] += value.item()
                self.progress.step()
                self.progress.new_test_values({
                    key: value / (b_i + 1) for key, value in metrics_sum.items()
                })
