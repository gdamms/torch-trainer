# torch-trainer
Simple custom pytorch trainer.


## Installation
```bash
pip install git+https://github.com/gdamms/torch-trainer.git
```


## Usage
```python
from trainer import train

train(
    model=model,  # torch.nn.Module
    train_loader=train_loader,  # torch.utils.data.DataLoader[torch.Tensor]
    epochs=...,  # int
    optimizer=...,  # torch.optim.Optimizer
    criterion=...,  # Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # Optional arguments
    metrics={"metric name": ...},  # Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    val_loader=valid_loader,  # torch.utils.data.DataLoader[torch.Tensor]
    test_loader=test_loader,  # torch.utils.data.DataLoader[torch.Tensor]
    epoch_callbacks=[...],  # List[Callable[[int, Dict[str, Any]], None]]
)
```


## Features
- When using the trainer, differrent progress bars are shown for training, validation and testing with their different
metrics.
- Loss and metrics are logged to the `runs` directory using
[tensorboard](https://pytorch.org/docs/stable/tensorboard.html).
  - If it is the first run of a model, a new directory is created.
  - If the model has already been trained by this trainer, the logs are appended to the existing directory.
