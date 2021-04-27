import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class SGCNDir(MessagePassing):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int,
        num_labels: int,
        gating: bool
        ):
        super().__init__(aggr='add')
        self.W_dir = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.b_lab = nn.Parameter(torch.FloatTensor(num_labels, dim_out))
        self.gating = gating

        if self.gating:
            self.W_dir_g = nn.Parameter(torch.FloatTensor(dim_in, 1))
            self.b_lab_g = nn.Parameter(torch.FloatTensor(num_labels, 1))

    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_label: torch.LongTensor
        ) -> torch.Tensor:
        # x has shape [N, dim_in]
        # edge_index has shape [2, E]
        # edge_label has shape [E]

        x = torch.matmul(x, self.W_dir)

        b_lab = torch.index_select(self.b_lab, 0, edge_label)
        b_lab_g = torch.index_select(self.b_lab_g, 0, edge_label) if self.gating else None
    
        return self.propagate(edge_index, x=x, b_lab=b_lab, b_lab_g=b_lab_g)

    def message(
        self, 
        x_j: torch.Tensor, 
        b_lab: torch.Tensor, 
        b_lab_g: torch.Tensor
        ) -> torch.Tensor:
        # x_j has shape [E, dim_out]
        # b_lab has shape [E, dim_out]
        # b_lab_g has shape [E, dim_out]

        print("x_j shape")
        print(x_j.shape)
        print("W_dir shape")
        print(self.W_dir.shape)

        x_j = torch.matmul(x_j, self.W_dir) + b_lab

        if self.gating: 
            gate = torch.sigmoid(torch.matmul(x_j, self.W_dir_g) + b_lab_g)
            x_j = gate * x_j
        
        return x_j

class SGCNLoop(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        gating: bool
        ):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.gating = gating
        
        # TODO: verify later
        if self.gating:
            self.lin_g = nn.Linear(dim_out, 1)
    
    def forward(self, x: int) -> torch.Tensor:
        x = self.lin(x)
        print("x pre gating dim")
        print(x.shape)

        if self.gating:
            gate = torch.sigmoid(self.lin_g(x))
            print("gate dim")
            print(gate.shape)
            x = gate * x
        
        print("sgcn loop dim")
        print(x.shape)
        return x
        
class SGCNConv(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_labels: int,
        gating: bool
        ):
        super().__init__()
        self.conv_loop = SGCNLoop(dim_in, dim_out, gating)
        self.conv_in = SGCNDir(dim_in, dim_out, num_labels, gating)
        self.conv_out = SGCNDir(dim_in, dim_out, num_labels, gating)

        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_label: torch.LongTensor
        ) -> torch.Tensor:
        # TODO: direction
        x_loop = self.conv_loop(x)
        print("conv_in x dim")
        print(x.shape)
        x_in = self.conv_in(x, edge_index, edge_label)
        print("conv_in x_in dim")
        print(x_in.shape)
        x_out = self.conv_out(x, torch.flip(edge_index, (-2, )), edge_label)
        return torch.relu(x_loop + x_in + x_out)