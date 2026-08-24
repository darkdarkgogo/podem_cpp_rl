import torch.nn as nn


_NORM_LAYER_FACTORY = {
    "batchnorm": nn.BatchNorm1d,
}

_ACT_LAYER_FACTORY = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(
        self,
        dim_in=256,
        dim_hidden=32,
        dim_pred=1,
        num_layer=3,
        norm_layer=None,
        act_layer=None,
        p_drop=0.5,
        sigmoid=False,
        tanh=False,
    ):
        super().__init__()
        if num_layer < 2:
            raise ValueError("num_layer should be at least 2")

        norm_ctor = _NORM_LAYER_FACTORY.get(norm_layer)
        act_ctor = _ACT_LAYER_FACTORY.get(act_layer)
        dropout_ctor = nn.Dropout if p_drop > 0 else None

        layers = [nn.Linear(dim_in, dim_hidden)]
        if norm_ctor:
            layers.append(norm_ctor(dim_hidden))
        if act_ctor:
            layers.append(act_ctor(inplace=True))
        if dropout_ctor:
            layers.append(dropout_ctor(p_drop))

        for _ in range(num_layer - 2):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_ctor:
                layers.append(norm_ctor(dim_hidden))
            if act_ctor:
                layers.append(act_ctor(inplace=True))
            if dropout_ctor:
                layers.append(dropout_ctor(p_drop))

        layers.append(nn.Linear(dim_hidden, dim_pred))
        if sigmoid:
            layers.append(nn.Sigmoid())
        if tanh:
            layers.append(nn.Tanh())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

