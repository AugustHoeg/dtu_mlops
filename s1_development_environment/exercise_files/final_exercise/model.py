import torch
from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, hidden_feats, p_drop) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_feats) - 1):
            self.layers.add_module(f"fc{i}", nn.Linear(hidden_feats[i], hidden_feats[i+1]))
            if i < len(hidden_feats) - 2:
                self.layers.add_module(f"relu{i}", nn.ReLU())
                self.layers.add_module(f"dropout{i}", nn.Dropout(p=p_drop))

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layers(x)
        x = F.log_softmax(x, dim=1)
        
        return x

if __name__ == "__main__":

    hidden_feats = [784, 512, 256, 128, 64, 10]
    p_drop = 0.2

    model = MyAwesomeModel(hidden_feats, p_drop)
    print(model)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")