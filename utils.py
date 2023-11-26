import torch
import torch.nn as nn

from dgl.nn import GraphConv, GATConv

class GCNHLayer(nn.Module):
    def __init__(self, S, in_dim, out_dim, K=3, norm_S=True, bias=True):
        super(GCNHLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        if norm_S:

            unnorm_S = S.clone()
            d = unnorm_S.sum(1)
            D_inv = torch.diag(torch.sqrt(1/d))
            D_inv[torch.isinf(D_inv)] = 0.

            S_norm = D_inv @ unnorm_S @ D_inv

            self.S = S_norm
        else:
            self.S = S
        self.N = self.S.shape[0]

        self.W = nn.Parameter(torch.empty(K, in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

        self.Spow = [torch.linalg.matrix_power(self.S, k) for k in range(self.K)]

    def forward(self, x): # Graph kept for compatibility
        assert (self.N, self.in_dim) == x.shape
        out = torch.zeros(self.N, self.out_dim, device=x.device)
        for k in range(self.K):
            #print(self.Spow[k].device, x.device, self.W.data.device)
            out += self.Spow[k] @ x @ self.W[k,:,:]
        if self.bias:
            return out + self.b[None,:]
        else:
            return out
            
class GCNH(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, act=nn.ReLU, bias=True, K=3, last_act=nn.LogSoftmax(dim=1), dropout=0., norm_S=True, batch_norm=True):
        super(GCNH, self).__init__()
        self.n_layers = n_layers
        assert self.n_layers > 1

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList()

        self.convs = nn.ModuleList()
        self.convs.append(GCNHLayer(S, in_dim, hid_dim, K, norm_S, bias))
        if self.batch_norm:
            self.batch_norm_layers.append(nn.BatchNorm1d(hid_dim))
        for i in range(self.n_layers-2):
            self.convs.append(GCNHLayer(S, hid_dim, hid_dim, K, norm_S, bias))
            if self.batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(hid_dim))
        self.convs.append(GCNHLayer(S, hid_dim, out_dim, K, norm_S, bias))

        self.nonlin = act()
        self.dropout = nn.Dropout(p=dropout)
        self.last_act_fn = last_act

    def forward(self, graph, h): # Graph kept for compatibility, although not used
        for i in range(self.n_layers-1):
            h = self.convs[i](h)
            h = self.nonlin(h)
            if self.batch_norm:
                h = self.batch_norm_layers[i](h)
            h = self.dropout(h)

        return self.convs[-1](h)
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.last_act_fn = nn.LogSoftmax(dim=1)

    def forward(self, graph, h): # Graph kept for compatibility, although not used
        h = self.layer1(h)
        h = self.nonlin(h)
        h = self.dropout(h)
        h = self.layer2(h)

        return h


class GCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., batch_norm=True):
        super(GCNN, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim)
        self.layer2 = GraphConv(hidden_dim, out_dim)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.last_act_fn = nn.LogSoftmax(dim=1)

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        if self.batch_norm:
            h = self.bn(h)
        h = self.nonlin(h)
        h = self.dropout(h)
        h = self.layer2(graph, h)
        #print(h.shape)
        #h = self.last_act_fn(h)
        #print(h.shape)
        return h
    

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, gat_params):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, **gat_params)
        self.nonlin = nn.ELU()

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.nonlin(h)
        h = self.layer2(graph, h)
        return h.squeeze()


def evaluate(features, graph, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def test_gcnh(model, graph, feat, label, train_mask, val_mask, test_mask, n_epochs, lr, wd, es_patience=-1, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.NLLLoss()

    best_acc_val = 0.
    best_acc_test = 0.
    best_iteration = 0
    count_es = 0

    loss_train, acc_train, acc_test = [torch.zeros(n_epochs) for _ in range(3)]

    for i in range(n_epochs):
        model.train()
        y_hat = model(graph, feat)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = loss_fn(y_hat[train_mask], label[train_mask])

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Compute accuracy on training/validation/test
        train_acc = evaluate(feat, graph, label, train_mask, model)
        val_acc = evaluate(feat, graph, label, val_mask, model)
        test_acc = evaluate(feat, graph, label, test_mask, model)

        loss_train[i] = loss.detach().cpu().item()
        acc_train[i] = train_acc
        acc_test[i] = test_acc

        if es_patience > 0:
            if val_acc > best_acc_val:
                count_es = 0
                best_iteration = i
                best_acc_val = val_acc
                best_acc_test = test_acc
            else:
                count_es += 1

            if count_es > es_patience:
                break

        if (i == 0 or (i+1) % 4 == 0) and verbose:
            print(f"Epoch {i+1}/{n_epochs} - Loss Train: {loss_train[i]} - Acc Train: {acc_train[i]} - Acc Val: {val_acc} - Acc Test: {acc_test[i]}", flush=True)

    return loss_train, acc_train, acc_test, best_acc_val, best_acc_test