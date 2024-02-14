import torch.nn as nn
import torch

import numpy as np

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

        M, Nin, Fin = X.shape
        Ns = S.shape[0]
        assert Nin == Ns
        assert Fin == self.in_dim

        X_out = X @ self.W[0,:,:]
        Sx = X
        for k in range(1, self.K):
            Sx = S @ Sx
            X_out += Sx @ self.W[k,:,:]

        if self.b is not None:
            return X_out + self.b[None,None,:]
        else:
            return X_out
        

class GFGCN(nn.Module):
    def __init__(self, S, in_dim, hid_dim, out_dim, n_layers, K, bias=True,
                 act=nn.ReLU, last_act=nn.Identity, dropout=0,
                 diff_layer=GFGCNLayer, norm_S=False, batch_norm=False, grad_S=True):
        super().__init__()
        self.S = nn.Parameter(S, requires_grad=grad_S)
        self.act = act()
        self.last_act = last_act()
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
        D_inv[torch.isnan(D_inv)] = 0. # Can happen if d is negative because of sqrt

        S_norm = D_inv @ unnorm_S @ D_inv

        self.S.data = S_norm

    def update_S(self, newS, normalize=True):
        self.S.data = newS
        if normalize:
            self.normalize_S()


## Non-convex Robust GNN Models

class RobustGNNModel:
    def __init__(self, S0, n_iters_H, lr, wd, lr_S, eval_freq, model_params, n_iters_out, n_iters_S,
                 problem_type="clas", loss_fn=nn.CrossEntropyLoss):
        self.lr = lr
        self.lr_S = lr_S
        self.wd = wd

        self.eval_freq = eval_freq

        self.n_iters_out = n_iters_out
        self.n_iters_H = n_iters_H
        self.n_iters_S = n_iters_S

        self.build_model(S0.clone(), model_params)

        self.problem_type = problem_type

        if self.problem_type == "clas":
            self.loss_fn = loss_fn()
            self.eval_fn = self.evaluate_clas
        elif self.problem_type == "reg":
            self.loss_fn = loss_fn()
            self.eval_fn = self.evaluate_reg

    def evaluate_clas(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            if len(labels) == 0:
                return 0.
            return correct.item() * 1.0 / len(labels)
        
    def evaluate_reg(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            return ((logits - labels)**2).mean()

    # Step W_H
    def stepH(self, x, labels, train_idx, val_idx, test_idx, gamma=1, S=None, verbose=False):
        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_iters_H) for _ in range(4)]

        for i in range(self.n_iters_H):
            self.model.train()
            y = self.model(x)
            loss = self.calc_loss(y[train_idx,:,:], labels[train_idx,:,:])

            self.opt_hW.zero_grad()
            loss.backward()
            self.opt_hW.step()

            # Compute loss on training/validation/test # TODO change name of variables
            acc_train[i] = self.eval_fn(x, labels, train_idx)
            acc_val[i] = self.eval_fn(x, labels, val_idx)
            acc_test[i] = self.eval_fn(x, labels, test_idx)
            losses[i] = loss.item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"\tEpoch (H) {i+1}/{self.n_iters_H} - Loss: {loss.item():.3f} - Train Acc: {acc_train[i]:.3f} - Test Acc: {acc_test[i]:.3f}")

        return acc_train, acc_val, acc_test, losses
    
    def stepS(self, Sn, x, labels, gamma, lambd, beta, alpha, omega, train_idx, S_true: np.ndarray=None, normalize_S: bool=False,
              test_idx=[], pct_S_pert: float=1., debug: bool=False):

        errs_S: np.array = np.zeros(self.n_iters_S)
        change_S: np.array = np.zeros(self.n_iters_S)
        norm_S: float = torch.linalg.norm(S_true)

        # self.model.train()

        lambd *= self.lr_S
        beta *= self.lr_S

        for i in range(self.n_iters_S):
            self.model.train()

            S_prev = self.model.S.data.clone()
            norm_S_orig = torch.linalg.norm(S_prev)

            S, loss = self.gradient_step_S(S=S_prev, x=x, y=labels, beta=beta, gamma=gamma, train_idx=train_idx, Sn=Sn, alpha=alpha, omega=omega)

            # Proximal for the distance to S_bar
            idxs_greater = torch.where(S - Sn > lambd)
            idxs_lower = torch.where(S - Sn < -lambd)
            S_prox = Sn.clone()
            S_prox[idxs_greater] = S[idxs_greater] - lambd
            S_prox[idxs_lower] = S[idxs_lower] + lambd

            # Considering the part of S that we know is correct, and taking that from Sn (if applicable)
            S_new = Sn.clone()
            N_mod = int(S.shape[0]*pct_S_pert)
            S_new[:N_mod,:N_mod] = S_prox[:N_mod,:N_mod]
            
            S = S_new

            # Projection onto \mathcal{S}
            S = torch.clamp(S, min=0., max=1.)
            # S = (S + S.T) / 2 # Not applicable for non-symmetric matrices

            errs_S[i] = torch.linalg.norm(S - S_true) / norm_S
            change_S[i] = torch.linalg.norm(S - S_prev) / norm_S_orig

            if debug and (i == 0 or (i+1) % self.eval_freq == 0):
                norm_A =  torch.linalg.norm(S)
                err_S2 = torch.linalg.norm(S/norm_A - S_true/norm_S)

                # Compute loss on training/validation/test # TODO change name of variables
                acc_train = self.eval_fn(x, labels, train_idx)
                acc_test = self.eval_fn(x, labels, test_idx)

                print(f'\tEpoch (S) {i+1}/{self.n_iters_S}: Loss: {loss:.2f}  - Train Acc: {acc_train:.2f} - Test Acc: {acc_test:.2f} - S-Sprev: {change_S[i]:.3f}  -  err_S: {errs_S[i]:.3f}  -  err_S (free scale): {err_S2:.3f}')

            self.model.update_S(S, normalize=normalize_S)
            
            # TODO: stopping criterion
        
        return errs_S, change_S

    def test_model(self, Sn, x, labels, gamma, lambd, beta, alpha=0., omega=0., # Alpha is only for l2 norm and elasticnet
                   train_idx=[], val_idx=[], test_idx=[],
                   norm_S=False, pct_S_pert=1., S_true=None, es_patience=-1, verbose=False,
                   debug_S=False, debug_H=False):

        accs_train = np.zeros((self.n_iters_out, self.n_iters_H))
        accs_test = np.zeros((self.n_iters_out, self.n_iters_H))
        errs_S = np.zeros((self.n_iters_out, self.n_iters_S))
        change_S = np.zeros((self.n_iters_out, self.n_iters_S))

        best_acc_val = np.inf
        best_acc_test = 0.
        best_err_S = 0.
        best_iteration = 0
        count_es = 0

        dec_sigma_count = 0

        for i in range(self.n_iters_out):
            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx,
                                                               test_idx, gamma, verbose=debug_H)

            # Graph estimation
            errs_S[i,:], change_S[i,:] = self.stepS(Sn, x, labels, gamma, lambd, beta, alpha, omega, train_idx,
                                                    S_true, normalize_S=norm_S, test_idx=test_idx,
                                                    pct_S_pert=pct_S_pert, debug=debug_S)
            if es_patience > 0:
                val_acc = self.eval_fn(x, labels, val_idx)
                if val_acc < best_acc_val:
                    count_es = 0
                    best_iteration = i
                    best_acc_val = val_acc
                    best_acc_test = self.eval_fn(x, labels, test_idx)
                    best_err_S = errs_S[i,-1]
                else:
                    count_es += 1

                if count_es > es_patience:
                    break

            if self.iters_sigma_dec > 0 and dec_sigma_count == self.iters_sigma_dec:
                self.sigma_i_idx += 1
                dec_sigma_count = 0
            else:
                dec_sigma_count += 1

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err S: {errs_S[i,-1]}")

        if es_patience <= 0:
            best_acc_val = self.eval_fn(x, labels, val_idx)
            best_acc_test = self.eval_fn(x, labels, test_idx)

        results_dict = {
            'accs_train': accs_train,
            'accs_test': accs_test,
            'best_iteration': best_iteration,
            'best_acc_val': best_acc_val,
            'best_acc_test': best_acc_test
        }

        S_dict = {
            'rec_S': self.model.S.data,
            'errs_S': errs_S,
            'change_S': change_S,
            'best_err_S': best_err_S
        }

        return results_dict, S_dict
    

###############################################################
###################### MODEL Score ############################
###############################################################
class RobustGNNScore(RobustGNNModel):
    def __init__(self, S0, n_iters_H, lr, wd, lr_S, eval_freq, model_params, n_iters_out, n_iters_S,
                 problem_type="clas", loss_fn=nn.CrossEntropyLoss, score_function=None, sigmas=[], sigma_i_idx = 3, dec_sigma=False):
        
        super().__init__(S0, n_iters_H, lr, wd, lr_S, eval_freq, model_params, n_iters_out, n_iters_S, problem_type, loss_fn)

        assert score_function is not None and sigma_i_idx >= 0 and sigma_i_idx < len(sigmas)
        self.score_function = score_function
        self.sigmas = sigmas
        
        if dec_sigma:
            self.sigma_i_idx = 0
            self.iters_sigma_dec = n_iters_out // len(sigmas)
        else:
            self.iters_sigma_dec = -1
            self.sigma_i_idx = sigma_i_idx

    def build_model(self, S, model_params):
        self.model = GFGCN(S=S, grad_S=True, **model_params)
        
        if model_params['bias']:
            self.opt_hW = torch.optim.Adam(
                [layer.W for layer in self.model.convs] + 
                [layer.b for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        else:
            self.opt_hW = torch.optim.Adam(
                [layer.W for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        
        self.opt_S = torch.optim.SGD([self.model.S], lr=self.lr_S)

    def calc_loss(self, y_hat, y_train):
        return self.loss_fn(y_hat, y_train)
    
    # Adding sparsity term
    def calc_loss_S(self, y_hat, y_train, beta=1, Sn=None, alpha=0):
        
        loss = self.loss_fn(y_hat, y_train)
        loss += beta*torch.sum(self.model.S)
        if Sn is not None and alpha > 0: # For L2 norm if needed
            loss += alpha*((self.model.S - Sn)**2).mean()
        return loss

    def gradient_step_S(self, S=None, x=None, y=None, beta=0., gamma=None, train_idx=None, Sn=None, alpha=0., omega=0.):
        y_hat = self.model(x)
        loss = self.calc_loss_S(y_hat[train_idx,:,:], y[train_idx,:,:], beta, Sn, alpha)

        self.opt_S.zero_grad()
        loss.backward()
        self.opt_S.step()

        S = self.model.S.data

        score = self.score_function(S, self.sigma_i_idx)
        # Remove padding
        score = score[:S.shape[0],:S.shape[0]]

        assert S.shape == score.shape

        alpha_l = omega * ((self.sigmas[self.sigma_i_idx]**2) / (self.sigmas[-1]**2))

        return S - alpha_l*score, loss.item()
