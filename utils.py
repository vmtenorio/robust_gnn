import torch
import torch.nn as nn
import numpy as np

import dgl


################
# DATA RELATED #
################
def pert_S(S, type="rewire", eps=0.1, creat=None, dest=None, sel_ratio=1, sel_node_idx=0, p_subset=0.5, n_p_white=0.):
    """
    Perturbate a given graph shift operator/adjacency matrix

    Assuming symmetry for every perturbation type but prob_nonsym

    There are two types of perturbation
    * prob: changes a value in the adjacency matrix with a certain
    probability. May result in denser graphs
    * rewire: rewire a percentage of original edges randomly
    """
    N = S.shape[0]

    if type == "prob":
        # Perturbated adjacency
        adj_pert_idx = np.triu(np.random.rand(N,N) < eps, 1)
        adj_pert_idx = adj_pert_idx + adj_pert_idx.T
        Sn = np.logical_xor(S, adj_pert_idx).astype(float)
    elif type == "prob_nonsym":
        # Perturbated adjacency
        adj_pert_idx = np.random.rand(N,N) < eps
        Sn = np.logical_xor(S, adj_pert_idx).astype(float)
    elif type == "rewire":
        # Edge rewiring
        idx_edges = np.where(np.triu(S,1) != 0)
        Ne = idx_edges[0].size
        unpert_edges = np.arange(Ne)
        for i in range(int(Ne*eps)):
            idx_modify = np.random.choice(unpert_edges)
             # To prevent modifying the same edge twice
            unpert_edges = np.delete(unpert_edges, np.where(unpert_edges == idx_modify))
            start = idx_edges[0][idx_modify]
            new_end = np.random.choice(np.delete(np.arange(N), start))
            idx_edges[0][idx_modify] = min(start, new_end)
            idx_edges[1][idx_modify] = max(start, new_end)
        Sn = np.zeros((N,N))
        Sn[idx_edges] = 1.
        assert np.all(np.tril(Sn) == 0)
        Sn = Sn + Sn.T + np.diag(np.diag(S))
    elif type == "rewire_nonsym":
        # Edge rewiring
        idx_edges = np.where(S != 0)
        Ne = idx_edges[0].size
        unpert_edges = np.arange(Ne)
        for i in range(int(Ne*eps)):
            idx_modify = np.random.choice(unpert_edges)
            # To prevent modifying the same edge twice
            unpert_edges = np.delete(unpert_edges, np.where(unpert_edges == idx_modify))
            start = idx_edges[0][idx_modify]
            new_end = np.random.choice(np.delete(np.arange(N), start))
            idx_edges[0][idx_modify] = start
            idx_edges[1][idx_modify] = new_end
        Sn = np.zeros((N,N))
        Sn[idx_edges] = 1.
    elif type == "creat-dest":

        creat = creat if creat is not None else eps
        dest = dest if dest is not None else eps

        A_x_triu = S.copy()
        A_x_triu[np.tril_indices(N)] = -1

        no_link_i = np.where(A_x_triu == 0)
        link_i = np.where(A_x_triu == 1)
        Ne = link_i[0].size

        # Create links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if no_link_i[0][i] < sel_node_idx or no_link_i[1][i] < sel_node_idx else 1 for i in range(no_link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(no_link_i[0].size) / no_link_i[0].size
        links_c = np.random.choice(no_link_i[0].size, int(Ne * creat),
                                replace=False, p=ps)
        idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

        # Destroy links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if link_i[0][i] < sel_node_idx or link_i[1][i] < sel_node_idx else 1 for i in range(link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(link_i[0].size) / link_i[0].size
        links_d = np.random.choice(link_i[0].size, int(Ne * dest),
                                replace=False, p=ps)
        idx_d = (link_i[0][links_d], link_i[1][links_d])

        A_x_triu[np.tril_indices(N)] = 0
        A_x_triu[idx_c] = 1.
        A_x_triu[idx_d] = 0.
        Sn = A_x_triu + A_x_triu.T
    elif type == "subset":
        N_mod = int(N*p_subset)
        adj_pert_idx = np.random.rand(N_mod,N_mod) < eps
        mask = np.zeros((N,N))
        mask[:N_mod,:N_mod] = adj_pert_idx
        Sn = np.logical_xor(S, mask).astype(float)
    else:
        raise NotImplementedError("Choose either prob, rewire or creat-dest perturbation types")
    
    if n_p_white > 0.:
        Nlinks = Sn.sum()
        Sn_all = Sn + n_p_white*np.linalg.norm(Sn)*np.random.randn(*Sn.shape) / np.sqrt(Nlinks)
        Sn = np.where(Sn == 0, Sn, Sn_all)

    return Sn


def get_data_dgl(dataset_name, verb=False, dev='cpu', idx=0):
    dataset = getattr(dgl.data, dataset_name)(verbose=False)

    g = dataset[0]

    # get graph and node feature
    S = g.adj().to_dense().numpy()
    feat = g.ndata['feat'].to(dev)

    # get labels
    label = g.ndata['label'].to(dev)
    n_class = dataset.num_classes

    # get data split
    masks = {}
    mask_labels = ['train', 'val', 'test']
    for lab in mask_labels:
        mask = g.ndata[lab + '_mask'].to(dev)
        # Select first data splid if more than one is available
        masks[lab] = mask[:,idx] if len(mask.shape) > 1 else mask
    
    if verb:
        N = S.shape[0]

        node_hom = dgl.node_homophily(g, g.ndata['label'])
        edge_hom = dgl.edge_homophily(g, g.ndata['label'])

        print('Dataset:', dataset_name)
        print(f'Number of nodes: {S.shape[0]}')
        print(f'Number of features: {feat.shape[1]}')
        print(f'Shape of signals: {feat.shape}')
        print(f'Number of classes: {n_class}')
        print(f'Norm of A: {np.linalg.norm(S, "fro")}')
        print(f'Max value of A: {np.max(S)}')
        print(f'Proportion of validation data: {torch.sum(masks["val"] == True).item()/N:.2f}')
        print(f'Proportion of test data: {torch.sum(masks["test"] == True).item()/N:.2f}')
        print(f'Node homophily: {node_hom:.2f}')
        print(f'Edge homophily: {edge_hom:.2f}')

    return S, feat, label, n_class, masks

##########
# Models #
##########
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
        self.layer1 = dgl.nn.GraphConv(in_dim, hidden_dim)
        self.layer2 = dgl.nn.GraphConv(hidden_dim, out_dim)
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
        self.layer1 = dgl.nn.GATConv(in_dim, hidden_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = dgl.nn.GATConv(hidden_dim * num_heads, out_dim, 1, **gat_params)
        self.nonlin = nn.ELU()

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.nonlin(h)
        h = self.layer2(graph, h)
        return h.squeeze()

####################
# Training Classes #
####################
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