from typing import *
import rich.progress


class TrainProgress(rich.progress.Progress):
    """A progress bar which tracks the progress of training epochs."""

    def __init__(
        self: 'TrainProgress',
        nb_epochs: int,
        train_size: int,
        val_size: int = 0,
        test_size: int = 0,
        *columns: str | rich.progress.ProgressColumn,
        **kwargs,
    ) -> None:
        """Initialize the progress bar.

        Args:
            nb_epochs (int): The number of epochs.
            train_size (int): The size of each tain epoch.
            val_size (int, optional): The size of each validation epoch. Defaults to 0.
            test_size (int, optional): The size of the test epoch. Defaults to 0.
            *columns (str | rich.progress.ProgressColumn): The columns to display.
            **kwargs: Additional arguments to pass to the parent class.
        """
        self.nb_epochs = nb_epochs
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        super().__init__(*columns, **kwargs)

        self.train_tasks = []
        self.val_tasks = []
        self.test_task = None
        self.total_task = self.add_task(
            "total",
            progress_type="total",
            total=nb_epochs * (train_size + val_size) + test_size,
        )
        self.train_values = []
        self.val_values = []
        self.test_values = {}

    def get_renderables(self: 'TrainProgress'):
        """Override the default renderables to display the epoch number."""
        pad = len(f"{self.nb_epochs}")
        for task_i, task in enumerate(self.tasks):
            # The total task.
            if task.fields.get("progress_type") == "total":
                self.columns = (
                    f"Working:",
                    rich.progress.BarColumn(),
                    f"{len(self.train_tasks):{pad}}/{self.nb_epochs}",
                    "•",
                    rich.progress.TimeRemainingColumn(),
                )

            # The train tasks.
            if task.fields.get("progress_type") == "train":
                epoch_id = task.fields.get("epoch_id")
                self.columns = (
                    f"Train {epoch_id:{pad}}:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.train_values[epoch_id-1].items()),
                )

            # The val tasks.
            if task.fields.get("progress_type") == "val":
                epoch_id = task.fields.get("epoch_id")
                self.columns = (
                    f"Val {epoch_id:{pad}}:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.val_values[epoch_id-1].items()),
                )

            # The test task.
            if task.fields.get("progress_type") == "test":
                self.columns = (
                    f"Test:",
                    rich.progress.BarColumn(),
                    f"{task.completed}/{task.total}",
                    "•",
                    rich.progress.TimeElapsedColumn(),
                    '•',
                    ' | '.join(
                        f"{key}: {value[-1]:.4f}" for key, value in self.test_values.items()),
                )

            yield self.make_tasks_table([task])

    def step_test(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) < self.nb_epochs:
            return False

        if self.tasks[self.train_tasks[-1]].completed < self.train_size:
            return False

        if self.val_size > 0:
            if len(self.val_tasks) < self.nb_epochs:
                return False

            if self.tasks[self.val_tasks[-1]].completed < self.val_size:
                return False

        if self.test_size == 0:
            return False

        if self.test_task is None:
            self.test_task = self.add_task(
                "Test",
                progress_type="test",
                total=self.test_size,
            )
            self.update(self.test_task, advance=count)
            self.update(self.total_task, advance=count)
            return True

        if self.test_task is not None:
            self.update(self.test_task, advance=count)
            self.update(self.total_task, advance=count)
            return True

    def step_val(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) == 0:
            return False

        if self.tasks[self.train_tasks[-1]].completed < self.train_size:
            return False

        if self.val_size == 0:
            return False

        if len(self.val_tasks) == 0 or (
            len(self.val_tasks) < self.nb_epochs
            and len(self.val_tasks) < len(self.train_tasks)
        ):
            self.val_values.append({})
            self.val_tasks.append(self.add_task(
                f"Val {len(self.val_tasks)+1}",
                progress_type="val",
                epoch_id=len(self.val_tasks)+1,
                total=self.val_size,
            ))
            self.update(self.val_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

        if self.tasks[self.val_tasks[-1]].completed < self.val_size:
            self.update(self.val_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

    def step_train(self: 'TrainProgress', count: int) -> bool:
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.

        Returns:
            bool: Whether step was successful.
        """
        if len(self.train_tasks) == 0 or self.tasks[self.train_tasks[-1]].completed == self.train_size:
            self.train_values.append({})
            self.train_tasks.append(self.add_task(
                f"Train {len(self.train_tasks)+1}",
                progress_type="train",
                epoch_id=len(self.train_tasks)+1,
                total=self.train_size,
            ))
            self.update(self.train_tasks[-1], advance=count)
            self.update(self.total_task, advance=count)
            return True

        self.update(self.train_tasks[-1], advance=count)
        self.update(self.total_task, advance=count)
        return True

    def step(self: 'TrainProgress', count: int = 1):
        """Advance the progress bar by the given number of steps.

        Args:
            count (int): The number of steps to advance the progress bar by.
        """
        if self.step_test(count):
            return

        if self.step_val(count):
            return

        if self.step_train(count):
            return

        raise RuntimeError("Progress bar already finished.")

    def new_train_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.train_values[-1].get(key, [])
            self.train_values[-1][key] = current_value + [value]

    def new_val_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.val_values[-1].get(key, [])
            self.val_values[-1][key] = current_value + [value]

    def new_test_values(self: 'TrainProgress', values: dict[str, Any]):
        """Update the progress bar with new values.

        Args:
            values (dict[str, Any]): The new values.
        """
        for key, value in values.items():
            current_value = self.test_values.get(key, [])
            self.test_values[key] = current_value + [value]
