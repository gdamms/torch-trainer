import os

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from typing import Callable, Iterable

import datetime
from .trainer_progress import TrainProgress
from .utils import set_model_attr, get_model_attr


def train(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    epochs: int,
    optimizer: Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    val_loader: DataLoader[torch.Tensor] | None = None,
    test_loader: DataLoader[torch.Tensor] | None = None,
    metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {},
    epoch_callbacks: Iterable[Callable[['Trainer'], None]] = [],
    save_chekpoint: int | bool = True,
):
    """This a function that trains a model.

    During training, the progress is displayed using progress bars. You can also find saved checkpoints in the `runs`
    directory. The `runs` directory is also used to store tensorboard logs. The tensorboard logs can be viewed by
    running `tensorboard --logdir=runs` in the terminal.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader[torch.Tensor]): The data loader for training.
        epochs (int): The number of epochs to train.
        optimizer (Optimizer): The optimizer to use.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
        val_loader (DataLoader[torch.Tensor] | None, optional): The data loader for validation.
        test_loader (DataLoader[torch.Tensor] | None, optional): The data loader for testing.
        metrics (dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The metrics to evaluate.
        epoch_callbacks (Iterable[Callable[[Trainer], None]], optional): The callbacks to run after each epoch.
        save_chekpoint (int | bool, optional): The number of epochs to save a checkpoint. If set to `0`, no checkpoint is saved
            but the last checkpoint is still saved. If set to `-1`, even the last checkpoint is not saved.
            (True == 1 and False == 0)
    """
    trainer = Trainer(
        model,
        train_loader,
        epochs,
        optimizer,
        criterion,
        val_loader,
        test_loader,
        metrics,
        epoch_callbacks,
        int(save_chekpoint),
    )
    trainer.start()


class Trainer:
    """A class which trains models."""

    # Directories.
    RUNS_DIR = 'runs'
    CHECKPOINTS_DIR = 'checkpoints'

    # Date format.
    DATE_FORMAT = '%Y%m%d-%H%M%S'

    # Model attributes.
    RUN_NAME = 'run_name'
    TRAINER_EPOCH = 'trainer_epoch'

    # Trainer works.
    UNKNOW_WORK = -1
    TRAIN_WORK = 0
    VALID_WORK = 1
    TEST_WORK = 2
    WORK_TAGS = {
        TRAIN_WORK: 'Train',
        VALID_WORK: 'Validation',
        TEST_WORK: 'Test',
    }

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[torch.Tensor],
        epochs: int,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_loader: DataLoader[torch.Tensor] | None = None,
        test_loader: DataLoader[torch.Tensor] | None = None,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {},
        epoch_callbacks: Iterable[Callable[['Trainer'], None]] = [],
        save_chekpoint: int = 1,
    ):
        """Initialize the trainer."""
        # Set the parameters.
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.epoch_callbacks = epoch_callbacks
        self.save_checkpoint = save_chekpoint

        # Set the run parameters.
        self.model_name = model.__class__.__name__
        self.date_time = datetime.datetime.now().strftime(Trainer.DATE_FORMAT)

        # Set the trainer epoch attributes.
        trainer_epoch = get_model_attr(model, Trainer.TRAINER_EPOCH)
        if trainer_epoch is None:
            self.trainer_epoch = 0
            set_model_attr(model, Trainer.TRAINER_EPOCH, str(self.trainer_epoch))
        else:
            self.trainer_epoch = int(trainer_epoch)
        self.epoch_start = self.trainer_epoch
        self.epoch_end = self.epoch_start + epochs

        # Set the run name and directories.
        self.run_name = get_model_attr(model, Trainer.RUN_NAME)
        if self.run_name is None:
            self.run_name = f'{self.date_time}_{self.model_name}'
            set_model_attr(model, Trainer.RUN_NAME, self.run_name)
        self.run_dir = os.path.join(self.RUNS_DIR, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.run_dir, Trainer.CHECKPOINTS_DIR)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize the tensorboard writer.
        self.writer = SummaryWriter(self.run_dir)

        # Initialize the progress bar.
        self.progress = TrainProgress(
            nb_epochs=epochs,
            train_size=len(train_loader),
            val_size=len(val_loader) if val_loader else 0,
            test_size=len(test_loader) if test_loader else 0,
        )

        # Initialize the trainer attributes.
        self.epoch_i = 0
        self.work = -1

    def start(self):
        """Start training the model."""
        with self.progress:
            for self.epoch_i in range(self.epoch_start + 1, self.epoch_end + 1):
                self.train()
                self.validate()

                self.trainer_epoch = self.epoch_i
                set_model_attr(self.model, 'trainer_epoch', str(self.trainer_epoch))

                # Save the model checkpoint.
                if self.save_current_epoch():
                    torch.save(self.model, f'runs/{self.run_name}/checkpoints/{self.trainer_epoch:04}e.pt')
                if self.save_checkpoint >= 0 and self.epoch_i == self.epoch_end:
                    torch.save(self.model, f'runs/{self.run_name}/checkpoints/last.pt')

                for callback in self.epoch_callbacks:
                    callback(self)

            self.test()

        self.writer.close()

    def train(self):
        """Train the model for one epoch."""
        self.work = Trainer.TRAIN_WORK
        self.model.train()
        for batch in self.train_loader:
            # Seprarate the inputs and labels.
            inputs = batch[:-1]
            labels = batch[-1]

            # Train the model.
            self.optimizer.zero_grad()
            output = self.model(*inputs)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # Update the progress bar.
            metrics_values = {'loss': loss.item()}
            metrics_values |= {f'{n}': m(output, labels).item() for n, m in self.metrics.items()}
            self.progress.step()
            self.progress.new_train_values(metrics_values)

        # Log the loss.
        work_tag = Trainer.WORK_TAGS[self.work]
        self.writer.add_scalar(f'Loss/{work_tag}', loss.item(), self.epoch_i)

    def evaluate(
        self,
        dataloader: DataLoader[torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ):
        """Evaluate the model.

        Args:
            dataloader (DataLoader): The data loader to evaluate on.
            metrics (dict): The metrics to evaluate.
        """
        self.model.eval()
        with torch.no_grad():
            metrics_sum = {name: 0.0 for name in metrics}
            for b_i, batch in enumerate(dataloader):
                # Seprarate the inputs and labels.
                inputs = batch[:-1]
                labels = batch[-1]

                # Evaluate the model.
                output = self.model(*inputs)
                metrics_values = {n: m(output, labels) for n, m in metrics.items()}

                # Update the progress bar.
                for n, v in metrics_values.items():
                    metrics_sum[n] += v.item()
                self.progress.step()
                self.progress.new_val_values({n: v / (b_i + 1) for n, v in metrics_sum.items()})

        # Log the metrics.
        for n, v in metrics_sum.items():
            work_tag = Trainer.WORK_TAGS[self.work]
            self.writer.add_scalar(f'{n}/{work_tag}', v / len(dataloader), self.epoch_i)

    def validate(self):
        """Validate the model."""
        self.work = Trainer.VALID_WORK
        if self.val_loader:
            self.evaluate(self.val_loader, {'Loss': self.criterion} | self.metrics)

    def test(self):
        """Test the model."""
        self.work = Trainer.TEST_WORK
        if self.test_loader:
            self.evaluate(self.test_loader, {'Loss': self.criterion} | self.metrics)

    def save_current_epoch(self):
        if self.save_checkpoint <= 0:
            return False
        return (self.epoch_i - self.epoch_start) % self.save_checkpoint == 0 \
                or self.epoch_i == self.epoch_end
