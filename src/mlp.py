import torch
from skorch import NeuralNetRegressor
from skorch.callbacks.lr_scheduler import LRScheduler
from torch.optim import SGD, Adam, AdamW


class MLPBase(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout, n_hidden, activation, use_batch_norm
    ):
        super().__init__()
        self.layers = []
        map_act_to_class = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "sigmoid": torch.nn.Sigmoid,
        }
        self.activation = map_act_to_class[activation]
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        self.layers.append(self.activation())
        for _ in range(n_hidden):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.activation())
            self.layers.append(torch.nn.Dropout(dropout))
            if use_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, 1))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, X):
        return self.model(X)


def create_mlp(config) -> NeuralNetRegressor:

    optimizer = config.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD

    lr_sched_step = config.pop("lr_sched_step", 0)
    lr_sched_gamma = config.pop("lr_sched_gamma", 0)
    callbacks = []
    if lr_sched_step:
        callbacks.append(
            (
                "lr_scheduler",
                LRScheduler(
                    policy=torch.optim.lr_scheduler.StepLR,  # type: ignore
                    step_size=lr_sched_step,
                    gamma=lr_sched_gamma,
                ),
            ),
        )
    net = NeuralNetRegressor(
        module=MLPBase,
        criterion=torch.nn.MSELoss,
        iterator_train__shuffle=True,
        callbacks=callbacks,
        **config
    )
    return net
