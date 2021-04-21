import torch
import torch.nn as nn

import numpy as np

class model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        x = np.ones((1,2))
        x = torch.from_numpy(x)
        x = x.cuda()
        print(x.get_device())


m1 = model()
m2 = model()

print("before cuda")
m1 = m1.cuda()

m1()
m2()

