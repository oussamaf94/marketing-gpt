import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.c_fc = nn.Linear(config["n_embed"], 4*config["n_embed"])
    self.c_proj=nn.Linear(4*config["n_embed"], config["n_embed"])
    self.drop= nn.Dropout(config["dropout"])

  def forward(self,x):
    x = self.c_fc(x)
    x = F.gelu(x)
    x = self.c_proj(x)
    x = self.drop(x)
    return x
