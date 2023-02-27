import torch
import torch.nn as nn

from example.qconfig import CustomQConfigs

class FooConv1x1(nn.Module):
    
    def __init__(self, set_qconfig):
        super().__init__()

        self.conv = nn.Conv2d(3, 3, 1, 1)   # 1x1 Conv kernel
        self.act = nn.ReLU()

        self.quant = torch.quantization.QuantStub(CustomQConfigs.get_default_qconfig())
        self.dequant = torch.quantization.DeQuantStub()

        self.modules_to_fuse = [['conv', 'act']]

        if set_qconfig:
            self.set_qconfig()
    
    def forward(self, x):
        x = self.quant(x)
        output_quant = self.act(self.conv(x))
        return self.dequant(output_quant) 

    def fuse(self):
        torch.ao.quantization.fuse_modules(self, self.modules_to_fuse, inplace=True)
        return self

    def set_qconfig(self):
        self.qconfig = CustomQConfigs.get_default_qconfig()
        return self
    
    def set_weights(self, multiplier):
        # Set bias to zero and conv weights to k*Identity
        self.conv.bias = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
        self.conv.weight = torch.nn.Parameter(multiplier * torch.eye(3).reshape(self.conv.weight.shape))


