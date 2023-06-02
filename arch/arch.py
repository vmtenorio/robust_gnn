import torch.nn as nn
import torch


class GCNLayerM1(nn.Module):
    def __init__(self, S, in_dim, out_dim, K, bias=True):
        super().__init__()
        self.S = S
        self.N = self.S.shape[1]
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.h = nn.Parameter(torch.empty((self.K)))
        torch.nn.init.constant_(self.h.data, 1.)
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None
        
    def forward(self, x):
        if x.ndim == 3:
            # Samples x Nodes x Features
            M, Nin, Fin = x.shape
            x = x.permute(1, 2, 0)
        else:
            # Nodes x Features
            Nin, Fin = x.shape
        assert Nin == self.N
        assert Fin == self.in_dim

        x_out = self.h[0] * x
        Sx = x
        for k in range(1, self.K):
            Sx = self.S @ Sx
            x_out += self.h[k] * Sx

        if self.b is not None:
            return x_out @ self.W + self.b[None,:]
        else:
            return x_out @ self.W

        # if x.ndim == 3:
        #     Sx = torch.zeros((self.K, M, Nin, Fin), device=x.device)
        # else:
        #     Sx = torch.zeros((self.K, Nin, Fin), device=x.device)
        # Sx[0,...] = x

        # for k in range(1, self.K):
        #     Sx[k,...] = self.S @ Sx[k-1,...].clone()

        # if x.ndim == 3:
        #     Hx = torch.sum(self.h[:,None,None,None] * Sx, 0)
        # else:
        #     Hx = torch.sum(self.h[:,None,None] * Sx, 0)

        # if self.b is not None:
        #     return Hx @ self.W + self.b[None,:]
        # else:
        #     return Hx @ self.W


class ComGCNLayer(nn.Module):
    def __init__(self, S, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # TODO: check other initializations
        # Init H
        self.H = nn.Parameter(torch.empty(S.shape))
        self.H.data.copy_(S)
        self.H.data += torch.eye(S.shape[0])
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, x):
        if self.b is not None:
            return self.H @ x @ self.W + self.b
        else:
            return self.H @ x @ self.W


##### V2 #####
class GCNModel1(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K, bias=True, nonlin=nn.ReLU,
                 last_nonlin=None):
        super().__init__()
        self.S = nn.Parameter(S)
        self.N = self.S.shape[0]
        self.K = K

        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin() if last_nonlin else None

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        ### SPECIFIC OF THE MODEL ###
        self.convs.append(GCNLayerM1(self.S, in_dim, hid_dim, self.K, bias))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayerM1(self.S, hid_dim, hid_dim, self.K, bias))
            self.convs.append(GCNLayerM1(self.S, hid_dim, out_dim, self.K, bias))
        #############################

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
        x = self.convs[-1](x)

        if self.last_nonlin:
            return self.last_nonlin(x)
        return x

    def update_S(self, newS):
        self.S.data = newS
        # for conv in self.convs:
        #     conv.S.data = newS


class ComGCNModel(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, bias=True, nonlin=nn.ReLU,
                 last_nonlin=None):
        super().__init__()

        assert n_layers > 1, 'n_layers must be at least 2'

        self.S = nn.Parameter(torch.empty(S.shape))
        self.S.data.copy_(S)
        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin() if last_nonlin else None
        self.n_layers = n_layers
        self.gcn = nn.ModuleList()

        self.gcn.append(ComGCNLayer(S, in_dim, hid_dim, bias))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.gcn.append(ComGCNLayer(S, hid_dim, hid_dim, bias))
            self.gcn.append(ComGCNLayer(S, hid_dim, out_dim, bias))

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.gcn[i](x))
        x = self.gcn[-1](x)

        if self.last_nonlin:
            return self.last_nonlin(x)
        return x

    # def update_S(self, newS):
    #     self.S = newS
    
    # def commutativity_term(self):
    #     commut = 0
    #     for layer in self.convs:
    #         commut += torch.linalg.norm(layer.H.data @ self.S - self.S @ layer.H.data, 'fro')**2
    #     return commut
