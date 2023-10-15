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

        if self.b is not None:
            return self.S @ x @ self.W + self.b[None,:]
        else:
            return self.S @ x @ self.W
        

class GCNLayerM12(nn.Module):
    def __init__(self, S, in_dim, out_dim, K, bias=True):
        super().__init__()
        self.S = S
        self.N = self.S.shape[1]
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.h = nn.Parameter(torch.empty((self.K)))
        torch.nn.init.constant_(self.h.data, 1./K)
        
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


##### V2 #####
class GCNLayerM2(nn.Module):
    def __init__(self, S, in_dim, out_dim, bias=True):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.H = nn.Parameter(torch.empty((self.N, self.N)))
        self.H.data = self.S + torch.eye(self.N, device=self.S.device)
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, x):
        if x.ndim == 3:
            _, Nin, Fin = x.shape
        else:
            Nin, Fin = x.shape
        assert Nin == self.N
        assert Fin == self.in_dim
        
        if self.b is not None:
            return self.H @ x @ self.W + self.b[None,:]
        else:
            return self.H @ x @ self.W


##### V2 #####
class GCNModel1(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K, bias=True, nonlin=nn.ReLU,
                 dropout=-1, last_nonlin=None):
        super().__init__()
        self.S = nn.Parameter(S)
        self.N = self.S.shape[0]
        self.K = K
        self.dropout = dropout

        self.normalize_S()

        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin() if last_nonlin else None

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        ### SPECIFIC OF THE MODEL ###
        self.convs.append(GCNLayerM1(self.S, in_dim, hid_dim, self.K, bias))
        if self.dropout > 0:
            self.dropout_layers.append(torch.nn.Dropout(dropout))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayerM1(self.S, hid_dim, hid_dim, self.K, bias))
                if self.dropout > 0:
                    self.dropout_layers.append(torch.nn.Dropout(dropout))
            self.convs.append(GCNLayerM1(self.S, hid_dim, out_dim, self.K, bias))
        #############################

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            if self.dropout > 0:
                x = self.dropout_layers[i](x)
        x = self.convs[-1](x)

        if self.last_nonlin:
            return self.last_nonlin(x)
        return x

    def update_S(self, newS, normalize=True):
        self.S.data = newS
        if normalize:
            self.normalize_S()
        # for conv in self.convs:
        #     conv.S.data = self.S.data.clone()

    def normalize_S(self):
        unnorm_S = self.S.data.clone()
        d = unnorm_S.sum(1)
        D_inv = torch.diag(torch.sqrt(1/d))
        D_inv[torch.isinf(D_inv)] = 0.

        S_norm = D_inv @ unnorm_S @ D_inv

        self.S.data = S_norm


##### V2 #####
class GCNModel2(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, bias=True, nonlin=nn.ReLU,
                 last_nonlin=None):
        super().__init__()
        self.S = S
        self.N = self.S.shape[0]

        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin() if last_nonlin else None

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        ### SPECIFIC OF THE MODEL ###
        self.convs.append(GCNLayerM2(self.S, in_dim, hid_dim, bias))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNLayerM2(self.S, hid_dim, hid_dim, bias))
            self.convs.append(GCNLayerM2(self.S, hid_dim, out_dim, bias))
        #############################


    def forward(self, x):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
        x = self.convs[-1](x)

        if self.last_nonlin:
            return self.last_nonlin(x)
        return x

    def update_S(self, newS, normalize=True):
        self.S = newS
    
    def commutativity_term(self):
        commut = 0
        for layer in self.convs:
            commut += torch.linalg.norm(layer.H.data @ self.S - self.S @ layer.H.data, 'fro')**2
        return commut



##### GFGCN


class GFGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias=True):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty((self.K, self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None
        
    def forward(self, X, S):

        Nin, Fin = X.shape
        Ns = S.shape[0]
        assert Nin == Ns
        assert Fin == self.in_dim

        X_out = X @ self.W[0,:,:]
        Sx = X
        for k in range(1, self.K):
            Sx = S @ Sx
            X_out += Sx @ self.W[k,:,:]

        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out
        

class GFGCN(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K, bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(), dropout=0,
                 diff_layer=GFGCNLayer):
        super().__init__()
        self.S = nn.Parameter(S)
        self.act = act
        self.last_act = last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias
        self.convs = nn.ModuleList()

        self.normalize_S()

        self.convs.append(diff_layer(in_dim, hid_dim, K, bias))
        if n_layers > 1:
            for _ in range(n_layers - 2):
                self.convs.append(diff_layer(hid_dim, hid_dim, K, bias))
            self.convs.append(diff_layer(hid_dim, out_dim, K, bias))

    def forward(self, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, self.S))
            X = self.dropout(X)
        X = self.convs[-1](X, self.S)
        return self.last_act(X)
    
    def normalize_S(self):
        unnorm_S = self.S.data.clone()
        d = unnorm_S.sum(1)
        D_inv = torch.diag(torch.sqrt(1/d))
        D_inv[torch.isinf(D_inv)] = 0.

        S_norm = D_inv @ unnorm_S @ D_inv

        self.S.data = S_norm

    def update_S(self, newS, normalize=True):
        self.S.data = newS
        if normalize:
            self.normalize_S()