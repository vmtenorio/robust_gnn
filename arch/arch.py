
import torch.nn as nn
import torch

class GCNLayerCoefs(nn.Module):
    def __init__(self, Spow, h, in_dim, out_dim):
        super().__init__()

        self.Spow = Spow
        self.N = self.Spow.shape[1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.h = h
        self.K = self.h.shape[0]
        assert self.Spow.shape[0] == self.K

        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

    def forward(self, x):
        if x.ndim == 3:
            _, Nin, Fin = x.shape
        else:
            Nin, Fin = x.shape
        assert Nin == self.N
        assert Fin == self.in_dim

        H = torch.sum(self.h[:,None,None]*self.Spow, 0)
        
        return H @ x @ self.W

class GCNCoefs(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]
        self.K = K

        self.h = nn.Parameter(torch.ones(self.K))

        self.Spow = self.calc_Spow(self.S)

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNLayerCoefs(self.Spow, self.h, in_dim, hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayerCoefs(self.Spow, self.h, hid_dim, hid_dim))
            self.convs.append(GCNLayerCoefs(self.Spow, self.h, hid_dim, out_dim))

    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = torch.tanh(self.convs[i](x))
        x = self.convs[-1](x)

        return x

    def calc_Spow(self, newS):
        Spow = torch.zeros((self.K, self.N, self.N), device=newS.device)
        Spow[0,:,:] = torch.eye(self.N, device=newS.device)
        for k in range(1, self.K):
            Spow[k,:,:] = Spow[k-1,:,:] @ newS
        return Spow

    def update_Spow(self, S):
        self.N = S.shape[0]
        self.S = S
        self.Spow = self.calc_Spow(S)
        for i in range(self.n_layers):
            self.convs[i].Spow = self.Spow
            self.convs[i].N = self.N

class GCNLayerCoefs_noW(nn.Module):
    def __init__(self, Spow, h):
        super().__init__()

        self.Spow = Spow
        self.N = self.Spow.shape[1]
        
        self.h = h
        self.K = self.h.shape[0]
        assert self.Spow.shape[0] == self.K

    def forward(self, x):
        Nin, _ = x.shape

        assert Nin == self.N

        H = torch.sum(self.h[:,None,None]*self.Spow, 0)
        
        return H @ x

class GCNCoefs_noW(nn.Module):
    def __init__(self, S, n_layers, K):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]
        self.K = K

        self.h = nn.Parameter(torch.ones(self.K))

        self.Spow = self.calc_Spow(self.S)

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNLayerCoefs_noW(self.Spow, self.h))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayerCoefs_noW(self.Spow, self.h))
            self.convs.append(GCNLayerCoefs_noW(self.Spow, self.h))

    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = torch.tanh(self.convs[i](x))
        x = self.convs[-1](x)

        return x

    def calc_Spow(self, newS):
        Spow = torch.zeros((self.K, self.N, self.N), device=newS.device)
        Spow[0,:,:] = torch.eye(self.N, device=newS.device)
        for k in range(1, self.K):
            Spow[k,:,:] = Spow[k-1,:,:] @ newS
        return Spow

    def update_Spow(self, S):
        self.calc_Spow(S)
        for i in range(self.n_layers):
            self.convs[i].Spow = self.Spow

class GCNLayernoW(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.N = self.H.shape[0]

    def forward(self, x):
        Nin, _ = x.shape

        assert Nin == self.N
        
        return self.H @ x

class GCNnoW(nn.Module):
    def __init__(self, S, n_layers):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]

        self.H = nn.Parameter(torch.empty(self.N, self.N))
        self.H.data = self.S

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNLayernoW(self.H))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayernoW(self.H))
            self.convs.append(GCNLayernoW(self.H))

    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = torch.tanh(self.convs[i](x))
        x = self.convs[-1](x)

        return x

class GCNLayer(nn.Module):
    def __init__(self, H, in_dim, out_dim):
        super().__init__()
        self.H = H
        self.N = self.H.shape[0]
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

    def forward(self, x):
        if x.ndim == 3:
            _, Nin, Fin = x.shape
        else:
            Nin, Fin = x.shape
        assert Nin == self.N
        assert Fin == self.in_dim
        
        return self.H @ x @ self.W

class GCN(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]

        self.H = nn.Parameter(torch.empty(self.N, self.N))
        self.H.data = self.S

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNLayer(self.H, in_dim, hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayer(self.H, hid_dim, hid_dim))
            self.convs.append(GCNLayer(self.H, hid_dim, out_dim))

    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = torch.tanh(self.convs[i](x))
        x = self.convs[-1](x)

        return x
