import math
import typing as ty

import numpy as np
import skorch
import torch
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger
from torch import Tensor, nn
from torch.optim import SGD, Adam, AdamW

import activations


class InputShapeSetterResnet(skorch.callbacks.Callback):
    def __init__(self, batch_size=None, categorical_indicator=None):
        self.categorical_indicator = categorical_indicator
        self.regression = True
        self.batch_size = batch_size

    def on_train_begin(self, net, X: np.ndarray, y=None, **kwargs):
        if self.categorical_indicator is None:
            d_numerical = X.shape[1]
            categories = None
        else:
            d_numerical = X.shape[1] - sum(self.categorical_indicator)
            categories = list((X[:, self.categorical_indicator].max(0) + 1).astype(int))
        net.set_params(
            module__d_numerical=d_numerical,
            module__categories=categories,
            module__d_out=2 if not self.regression else 1,
        )


def create_resnet_skorch(wandb_run=None, categorical_indicator=None, **kwargs):
    if "verbose" not in kwargs:
        verbose = 0
    else:
        verbose = kwargs.pop("verbose")
    if "es_patience" not in kwargs:
        es_patience = 40
    else:
        es_patience = kwargs.pop("es_patience")
    optimizer = kwargs.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    batch_size = kwargs.pop("batch_size")
    callbacks = [
        InputShapeSetterResnet(categorical_indicator=categorical_indicator),
        EarlyStopping(monitor="valid_loss", patience=es_patience),
    ]

    lr_sched_step = kwargs.pop("lr_sched_step", 0)
    lr_sched_gamma = kwargs.pop("lr_sched_gamma")
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
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    mlp_skorch = NeuralNetRegressor(
        ResNet,
        criterion=torch.nn.MSELoss,
        optimizer=optimizer,
        batch_size=max(batch_size, 1),
        iterator_train__shuffle=True,
        module__d_numerical=1,
        module__categories=None,
        module__d_out=1,
        module__regression=True,
        module__categorical_indicator=categorical_indicator,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs
    )
    return mlp_skorch


class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        super().__init__()

        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
                normalization
            ](d)

        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.main_activation = activations.get_activation_fn(activation)
        self.last_activation = activations.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(
                            d, d_hidden * (2 if activation.endswith("glu") else 1)
                        ),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x) -> Tensor:
        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
        else:
            x_num = x
        x = []
        if x_num is not None:
            x.append(x_num)
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer["linear1"](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x
