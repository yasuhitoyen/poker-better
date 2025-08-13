import torch
import torch.nn as nn
class FCNRewardFunction(nn.Module):
  def __init__(self,
               input_dim:int,
               hidden_dims:List[int] = [256, 128, 64]):
    super().__init__()
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
      layers.extend([
          nn.Linear(in_features=prev_dim,
                    out_features=hidden_dim),
          nn.ReLU(),
          nn.Dropout(p=0.1)
      ])
      prev_dim=hidden_dim

    # Add final layer to output single reward value
    layers.append(nn.Linear(in_features=prev_dim, out_features=1))

    self.network = nn.Sequential(*layers)

  def forward(self, x):
      return self.network(x)
