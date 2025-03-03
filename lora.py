import torch
from torch import nn

#Weight matrix with size (d, k)
#Matrix A size = (r, k)
#Matrix B size = (d, r)

class LoRALayer(nn.Module):
    def __init__(self, layer: nn.Linear, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.layer = layer

        self.d = self.layer.weight.shape[0]
        self.k = self.layer.weight.shape[1]
        self.scaling_factor = self.alpha / self.rank

        self.A = nn.Parameter(torch.randn(self.rank, self.k), requires_grad=True) #Learned matrix set to gaussian distribution
        self.B = nn.Parameter(torch.zeros(self.d, self.rank), requires_grad=True) #Learned matrix set to all zeros


    def forward(self, x):
        self.lora_update = torch.matmul(self.B, self.A)

        updated_weights = self.layer(x) + self.scaling_factor * (x @ self.lora_update)
        return updated_weights



        



        
    


    


