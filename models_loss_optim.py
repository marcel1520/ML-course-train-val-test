import torch
from torch import nn
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"
device


class MoonModelV1(nn.Module):
    def __init__(self, input_features, output_features, hidden_layer):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer, out_features=output_features)
        )
        
    def forward(self, x):
        return self.layer_stack(x)


class MoonModelV2(nn.Module):
    def __init__(self, input_features, output_features, hidden_layer):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.Tanh(),
            nn.Linear(in_features=hidden_layer, out_features=output_features)
        )
        
    def forward(self, x):
        return self.layer_stack(x)


class MoonModelV3(nn.Module):
    def __init__(self, input_features, output_features, hidden_layer):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(in_features=hidden_layer, out_features=output_features)
        )
        
    def forward(self, x):
        return self.layer_stack(x)
    

moon_model_V1_1 = MoonModelV1(2, 1, 16).to(device)
moon_model_V1_2 = MoonModelV1(2, 1, 128).to(device)


moon_model_V2_1 = MoonModelV2(2, 1, 16).to(device)
moon_model_V2_2 = MoonModelV2(2, 1, 128).to(device)


moon_model_V3_1 = MoonModelV3(2, 1, 16).to(device)
moon_model_V3_2 = MoonModelV3(2, 1, 128).to(device)


# ------------------------------------------------


loss_fn_1 = nn.BCEWithLogitsLoss()
loss_fn_2 = nn.CrossEntropyLoss()


# ------------------------------------------------


def optimizer_setup_SGD(model, learning_rate):
    sgd_optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=learning_rate)
    return sgd_optimizer

def optimizer_setup_Adam(model, learning_rate):
    adam_optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    return adam_optimizer