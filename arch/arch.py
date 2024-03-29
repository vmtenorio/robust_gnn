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
                 act=nn.ReLU, last_act=nn.Identity, dropout=0,
                 diff_layer=GFGCNLayer, norm_S=False, batch_norm=False):
        super().__init__()
        self.S = nn.Parameter(S)
        self.act = act()
        self.last_act = last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList()

        if norm_S:
            self.normalize_S()

        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(diff_layer(in_dim, hid_dim, K, bias))
            if self.batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(hid_dim))
            for _ in range(n_layers - 2):
                self.convs.append(diff_layer(hid_dim, hid_dim, K, bias))
                if self.batch_norm:
                    self.batch_norm_layers.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(diff_layer(hid_dim, out_dim, K, bias))
        else:
            self.convs.append(diff_layer(in_dim, out_dim, K, bias))

    def forward(self, X):
        for i in range(self.n_layers - 1):
            X = self.convs[i](X, self.S)
            X = self.act(X)
            if self.batch_norm:
                X = self.batch_norm_layers[i](X)
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



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h):
        h = self.layer1(h)
        h = self.nonlin(h)
        h = self.dropout(h)
        h = self.layer2(h)
        return h
