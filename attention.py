import torch
from torch import nn, optim

class SelAttention(nn.Module):
    def __init__(self, inp_dim, out_dim, bias=False):
        super(SelAttention, self).__init__()

        self.wq = nn.Linear(inp_dim, out_dim, bias=bias)
        self.wk = nn.Linear(inp_dim, out_dim, bias=bias)
        self.wv = nn.Linear(inp_dim, out_dim, bias=bias)

    def forward(self, x):

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        s = q @ v.transpose(1, 2)
        w = torch.softmax(s/k.shape[-1]**0.5, dim=-1)
        cv = w @ v
        return cv


x = torch.randn(2, 10, 256)
sa = SelAttention(x.shape[-1], x.shape[-1])
print(sa(x).shape)

