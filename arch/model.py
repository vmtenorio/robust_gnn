import torch.nn as nn
import torch

import numpy as np
from arch.arch import GCNModel1, GFGCN

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
            logits = self.model(features).squeeze()
            logits = logits[mask]
            labels = labels[mask]
            return ((logits - labels)**2).mean()

    # Step W_H
    def stepH(self, x, labels, train_idx, val_idx, test_idx, gamma=1, S=None, verbose=False):
        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_iters_H) for _ in range(4)]

        for i in range(self.n_iters_H):
            self.model.train()
            y = self.model(x).squeeze()
            loss = self.calc_loss(y[train_idx], labels[train_idx], gamma)

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
    
    def stepS(self, Sn, x, labels, gamma, lambd, beta, alpha, train_idx, S_true: np.ndarray=None, normalize_S: bool=False,
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

            S, loss = self.gradient_step_S(S=S_prev, x=x, y=labels, beta=beta, gamma=gamma, train_idx=train_idx, Sn=Sn, alpha=alpha)

            # Proximal for the distance to S_bar
            idxs_greater = torch.where(S - Sn > lambd)
            idxs_lower = torch.where(S - Sn < -lambd)
            S_prox = Sn.clone()
            S_prox[idxs_greater] = S[idxs_greater] - lambd
            S_prox[idxs_lower] = S[idxs_lower] + lambd

            # Considering the part of S that we know is correct, and taking that from Sn
            S_new = Sn.clone()
            N_mod = int(S.shape[0]*pct_S_pert)
            S_new[:N_mod,:N_mod] = S_prox[:N_mod,:N_mod]
            
            S = S_new

            # Projection onto \mathcal{S}
            S = torch.clamp(S, min=0., max=1.)
            # S = (S + S.T) / 2 # Not suitable for non-symmetric matrices

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

    def test_model(self, Sn, x, labels, gamma, lambd, beta, alpha=0., # Alpha is only for l2 norm and elasticnet
                   train_idx=[], val_idx=[], test_idx=[],
                   norm_S=False, pct_S_pert=1., S_true=None, es_patience=-1, verbose=False,
                   debug_S=False, debug_H=False):

        accs_train = np.zeros((self.n_iters_out, self.n_iters_H))
        accs_test = np.zeros((self.n_iters_out, self.n_iters_H))
        errs_S = np.zeros((self.n_iters_out, self.n_iters_S))
        change_S = np.zeros((self.n_iters_out, self.n_iters_S))

        best_acc_val = 0.
        best_acc_test = 0.
        best_err_S = 0.
        best_iteration = 0
        count_es = 0

        for i in range(self.n_iters_out):
            # TODO: separate step for H and W
            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx,
                                                               test_idx, gamma, verbose=debug_H)

            new_S = self.model.S.data.clone()
            # Graph estimation
            errs_S[i,:], change_S[i,:] = self.stepS(Sn, x, labels, gamma, lambd, beta, alpha, train_idx,
                                                    S_true, normalize_S=norm_S, test_idx=test_idx,
                                                    pct_S_pert=pct_S_pert, debug=debug_S)
            
            if es_patience > 0:
                val_acc = self.eval_fn(x, labels, val_idx)
                if val_acc > best_acc_val:
                    count_es = 0
                    best_iteration = i
                    best_acc_val = val_acc
                    best_acc_test = self.eval_fn(x, labels, test_idx)
                    best_err_S = errs_S[i,-1]
                else:
                    count_es += 1

                if count_es > es_patience:
                    break

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err S: {errs_S[i,-1]}")

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
######################## MODEL 1 ##############################
###############################################################
class RobustGNNModel1(RobustGNNModel):

    def build_model(self, S, model_params):
        #self.model = GCNModel1(S=S, **model_params)
        self.model = GFGCN(S=S, **model_params)
        
        if model_params['bias']:
            self.opt_hW = torch.optim.Adam(
                #[layer.h for layer in self.model.convs] +
                [layer.W for layer in self.model.convs] + 
                [layer.b for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        else:
            self.opt_hW = torch.optim.Adam(
                #[layer.h for layer in self.model.convs] +
                [layer.W for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        
        self.opt_S = torch.optim.SGD([self.model.S], lr=self.lr_S)

    # NOTE: gamma added to keep using same function with models 1 and 2
    def calc_loss(self, y_hat, y_train, gamma=None):
        return self.loss_fn(y_hat, y_train)
    
    # Adding sparsity term
    def calc_loss_S(self, y_hat, y_train, beta=1, Sn=None, alpha=0):
        
        loss = self.loss_fn(y_hat, y_train)
        loss += beta*torch.sum(self.model.S)
        if Sn is not None and alpha > 0: # For L2 norm if needed
            loss += alpha*((self.model.S - Sn)**2).mean()
        return loss


    def gradient_step_S(self, S=None, x=None, y=None, beta=1, gamma=None, train_idx=None, Sn=None, alpha=0.):
        y_hat = self.model(x).squeeze()
        loss = self.calc_loss_S(y_hat[train_idx], y[train_idx], beta, Sn, alpha)

        self.opt_S.zero_grad()
        loss.backward()
        self.opt_S.step()

        return self.model.S.data, loss.item()
