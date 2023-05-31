import logging
import torch.nn as nn
import torch

import numpy as np
from arch.arch import GCNModel1, GCNModel2

from sev_filters_opt import graph_id, graph_id_rew

## Non-convex Robust GNN Models

class RobustGNNModel:
    def __init__(self, S0, n_iters_H, lr, wd, lr_S, eval_freq, model_params, n_iters_out, n_iters_S,
                 problem_type="clas", reduct='mean'):
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
            self.loss_fn = nn.CrossEntropyLoss(reduction=reduct)
            # self.loss_fn = nn.NLLLoss(reduction=reduct)
        elif self.problem_type == "reg":
            self.loss_fn = nn.MSELoss(reduction=reduct)

    def evaluate_clas(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
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

            if self.problem_type == "clas":
                eval_fn = self.evaluate_clas
            elif self.problem_type == "reg":
                eval_fn = self.evaluate_reg
            # Compute loss on training/validation/test # TODO change name of variables
            acc_train[i] = eval_fn(x, labels, train_idx)
            acc_val[i] = eval_fn(x, labels, val_idx)
            acc_test[i] = eval_fn(x, labels, test_idx)
            losses[i] = loss.item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"\tEpoch (H) {i+1}/{self.n_iters_H} - Loss: {loss.item():.3f} - Train Acc: {acc_train[i]:.3f} - Test Acc: {acc_test[i]:.3f}")

        return acc_train, acc_val, acc_test, losses
    
    ##### STEP for model 1 #####
    def stepS(self, Sn, x, labels, gamma, lambd, beta, train_idx, S_true=None,
              test_idx=[], debug=False):
        errs_S = np.zeros(self.n_iters_S)
        change_S = np.zeros(self.n_iters_S)
        norm_S = torch.linalg.norm(S_true)

        # self.model.train()

        lambd *= self.lr_S
        beta *= self.lr_S

        for i in range(self.n_iters_S):
            self.model.train()

            S_prev = self.model.S.data.clone()
            norm_S_orig = torch.linalg.norm(S_prev)

            S, loss = self.gradient_step_S(x, labels, beta, gamma, train_idx)

            # Proximal for the distance to S_bar
            idxs_greater = torch.where(S - Sn > lambd)
            idxs_lower = torch.where(S - Sn < -lambd)
            S_prox = Sn.clone()
            S_prox[idxs_greater] = S[idxs_greater] - lambd
            S_prox[idxs_lower] = S[idxs_lower] + lambd
            S = S_prox

            # Projection onto \mathcal{S}
            S = torch.where(S < 0., 0., S)
            S = torch.where(S > 1., 1., S)
            S = (S + S.T) / 2

            errs_S[i] = torch.linalg.norm(S - S_true) / norm_S
            change_S[i] = torch.linalg.norm(S - S_prev) / norm_S_orig

            if debug and (i == 0 or (i+1) % self.eval_freq == 0):
                norm_A =  torch.linalg.norm(S)
                err_S2 = torch.linalg.norm(S/norm_A - S_true/norm_S)

                if self.problem_type == "clas":
                    eval_fn = self.evaluate_clas
                elif self.problem_type == "reg":
                    eval_fn = self.evaluate_reg
                # Compute loss on training/validation/test # TODO change name of variables
                acc_train = eval_fn(x, labels, train_idx)
                acc_test = eval_fn(x, labels, test_idx)

                print(f'\tEpoch (S) {i+1}/{self.n_iters_S}: Loss: {loss:.2f}  - Train Acc: {acc_train:.2f} - Test Acc: {acc_test:.2f} - S-Sprev: {change_S[i]:.3f}  -  err_S: {errs_S[i]:.3f}  -  err_S (free scale): {err_S2:.3f}')

            self.model.update_S(S)
            
            # TODO: stopping criterion
        
        return errs_S, change_S


    def test_model(self, Sn, x, labels, gamma, lambd, beta, train_idx=[], val_idx=[], test_idx=[],
                   S_true=None, verbose=False, debug_S=False, debug_H=False):

        accs_train = np.zeros((self.n_iters_out, self.n_iters_H))
        accs_test = np.zeros((self.n_iters_out, self.n_iters_H))
        errs_S = np.zeros((self.n_iters_out, self.n_iters_S))
        change_S = np.zeros((self.n_iters_out, self.n_iters_S))

        for i in range(self.n_iters_out):
            # TODO: separate step for H and W
            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx,
                                                               test_idx, gamma, verbose=debug_H)

            # Graph estimation
            errs_S[i,:], change_S[i,:] = self.stepS(Sn, x, labels, gamma, lambd, beta, train_idx,
                                                    S_true, test_idx=test_idx, debug=debug_S)

            if verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err S: {errs_S[i,-1]}")

        return accs_train, accs_test, self.model.S.data, errs_S, change_S
    

###############################################################
######################## MODEL 1 ##############################
###############################################################
class RobustGNNModel1(RobustGNNModel):

    def build_model(self, S, model_params):
        self.model = GCNModel1(S=S, **model_params)
        
        if model_params['bias']:
            self.opt_hW = torch.optim.Adam(
                [layer.h for layer in self.model.convs] +
                [layer.W for layer in self.model.convs] + 
                [layer.b for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        else:
            self.opt_hW = torch.optim.Adam(
                [layer.h for layer in self.model.convs] +
                [layer.W for layer in self.model.convs],
                lr=self.lr, weight_decay=self.wd)
        
        self.opt_S = torch.optim.SGD([self.model.S], lr=self.lr_S)

    # NOTE: gamma added to keep using same function with models 1 and 2
    def calc_loss(self, y_hat, y_train, gamma=None):
        return self.loss_fn(y_hat, y_train)
    
    # Adding sparsity term
    def calc_loss_S(self, y_hat, y_train, beta=1):
        return self.loss_fn(y_hat, y_train) + beta*torch.sum(self.model.S)


    def gradient_step_S(self, x, y, beta=1, gamma=None, train_idx=None):
        y_hat = self.model(x).squeeze()
        loss = self.calc_loss_S(y_hat[train_idx], y[train_idx], beta)

        self.opt_S.zero_grad()
        loss.backward()
        self.opt_S.step()

        return self.model.S.data, loss.item()
        
###############################################################
######################## MODEL 2 ##############################
###############################################################
class RobustGNNModel2(RobustGNNModel):
    def build_model(self, S, model_params):
        self.model = GCNModel2(S=S, **model_params)

        self.opt_hW = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def calc_loss(self, y_hat, y_train, gamma=1):
        return self.loss_fn(y_hat, y_train) + gamma*self.model.commutativity_term()
    

    # NOTE: Assume S=A
    def gradient_step_S(self, S, gamma=1, beta=1, x=None, y=None, train_idx=None):
        grad = 0
        for layer in self.model.convs:
            H = layer.H.data
            # Gradient corresponding to commutativity || HA-AH ||_F^2
            grad += 2*gamma*(H.T @ H @ S - H.T @ S @ H - H @ S @ H.T + S @ H @ H.T)
            # S -= 2*gamma*(H.T @ H @ S - H.T @ S @ H - H @ S @ H.T + S @ H @ H.T)

        # Gradient corresponding to sparsity of A
        grad += beta*torch.ones(S.shape, device=S.device)
        return S - self.lr_S*grad
    


###############################################################
####################### MY MODEL 2 ############################
###############################################################
# TODO: distinguir entre Sn y S_init
# TODO: separar update H y W
class RobustGNNModel2_v2(RobustGNNModel):
    def build_model(self, S, model_params):
        self.model = GCNModel2(S=S, **model_params)
        self.opt_W = torch.optim.Adam([layer.W for layer in self.model.convs],
                                      r=self.lr, weight_decay=self.wd)
        self.opt_H = torch.optim.Adam([layer.H for layer in self.model.convs],
                                      lr=self.lr, weight_decay=self.wd)

    def calc_loss(self, y_hat, y_train, gamma=1):
        return self.loss_fn(y_hat, y_train) + gamma*self.model.commutativity_term()
        



###############################################################
############## Convex alternatives to estimate S ##############
###############################################################
class ModelCVX:
    def __init__(self, S, n_epochs, lr, wd, eval_freq, model_params):

        self.n_iters_H = n_epochs
        self.lr = lr
        self.wd = wd

        self.eval_freq = eval_freq

        self.loss_fn = nn.MSELoss()

        self.build_model(S, model_params)

    def build_model(self, S, model_params):
        return

    def predict(self, x):
        return self.model(x)
    
    def evaluate_reg(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features).squeeze()
            logits = logits[mask]
            labels = labels[mask]
            return ((logits - labels)**2).mean()

    def test_reg(self, S, X, Y, train_idx=[], val_idx=[], test_idx=[], verbose=True):

        loss_train, loss_test = [np.zeros(self.n_iters_H) for _ in range(2)]

        for i in range(self.n_iters_H):
            self.model.train()
            y_hat = self.model(X).squeeze()
            loss = self.calc_loss(y_hat[train_idx], Y[train_idx], S)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_train[i] = loss.cpu().item()
            loss_test[i] = self.evaluate_reg(X, Y, test_idx)

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_iters_H} - Loss train: {loss_train[i]} - Loss: {loss_test[i]}", flush=True)

        return loss_train, loss_test

    def test_clas(self, x, labels, train_idx=[], val_idx=[], test_idx=[], S=None, verbose=True):

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        loss_fn = nn.CrossEntropyLoss()

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_iters_H) for _ in range(4)]

        for i in range(self.n_iters_H):
            y = self.model(x)
            loss = self.calc_loss(y[train_idx], labels[train_idx], S)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = torch.argmax(y, 1)
            results = (preds == labels).type(torch.float32)
            acc_train[i] = results[train_idx].mean().item()
            acc_val[i] = results[val_idx].mean().item()
            acc_test[i] = results[test_idx].mean().item()
            losses[i] = loss.item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_iters_H} - Loss: {loss.item()} - Train Acc: {acc_train[i]} - Test Acc: {acc_test[i]}", flush=True)

        return acc_train, acc_val, acc_test, losses

    def test_iterative_clas(self, Sn, x, labels, lambd, gamma, n_iters, train_idx=[], val_idx=[], test_idx=[]):

        S_id = Sn

        accs_train = np.zeros((n_iters, self.n_iters_H))
        accs_test = np.zeros((n_iters, self.n_iters_H))

        for i in range(n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            S_gcn = torch.Tensor(S_id).to(x.device)

            self.update_S(S_gcn)
            
            # Filter estimation
            accs_train[i,:], _, accs_test[i,:], _ = self.test_clas(S_gcn, x, labels, gamma, train_idx, val_idx, test_idx, verbose=False)

            h_id = self.model.h.data
            H_id = torch.sum(h_id[:,None,None]*self.model.Spow, 0).cpu().numpy()

            #print("Graph identification")
            # Graph estimation
            S_id = graph_id(Sn, H_id, np.zeros(Sn.shape), lambd, gamma, 0.)

        return accs_train, accs_test, S_id

    def test_iterative_reg(self, Sn, x, labels, lambd, gamma, n_iters, train_idx=[], val_idx=[], test_idx=[], S_true=None, Cy=None, verbose=False):

        S_id = Sn.cpu().numpy()
        S_true = S_true.cpu().numpy()

        loss_train = np.zeros((n_iters, self.n_iters_H))
        loss_test = np.zeros((n_iters, self.n_iters_H))
        errs_S = np.zeros((n_iters))

        norm_S = np.linalg.norm(S_true)

        for i in range(n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            S_gcn = torch.Tensor(S_id).to(x.device)

            change_S = np.linalg.norm(self.model.S.data.cpu().numpy() - S_id)

            self.update_S(S_gcn)
            
            # Filter estimation
            loss_train[i,:], loss_test[i,:] = self.test_reg(S_gcn, x, labels, train_idx, val_idx, test_idx, verbose=False)

            Hs = self.build_filters()

            #print("Graph identification")
            # Graph estimation
            S_id = graph_id(Sn.cpu().numpy(), Hs, np.zeros(Hs.shape) if Cy is None else Cy, lambd, gamma, 10.)

            errs_S[i] = np.linalg.norm(S_id - S_true) / norm_S

            if verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {loss_test[i,-1]} - Err S: {errs_S[i]} - Change S: {change_S}")

        return loss_train, loss_test, S_id
    

class ModelCVX1(ModelCVX):
    def build_model(self, S, model_params):
        self.model = GCNModel1(S=S.clone(), **model_params)

        self.opt = torch.optim.Adam(
            [layer.h for layer in self.model.convs] + [layer.W for layer in self.model.convs],
            lr=self.lr, weight_decay=self.wd)
        
        self.model.S.requires_grad = False
        for conv in self.model.convs:
            conv.S.requires_grad = False

    def update_S(self, newS):
        self.model.update_Spow(newS)
    
    def calc_loss(self, y_hat, y_train, S=None):
        return self.loss_fn(y_hat, y_train)
    
    def build_filters(self):
        Hs = []
        for conv in self.model.convs:
            h_id = conv.h.data
            Hs.append(torch.sum(h_id[:,None,None]*self.model.Spow, 0).cpu().numpy())
        return np.array(Hs)

class ModelCVX2(ModelCVX):
    def build_model(self, S, model_params):
        self.model = GCNModel2(S=S.clone(), **model_params)

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.wd)
    
    def update_S(self, newS):
        return # Not needed, as we are dealing with H
    
    def calc_loss(self, y_hat, y_train, S=None):
        return self.loss_fn(y_hat, y_train) + self.model.commutativity_term(S)
    
    def build_filters(self):
        Hs = []
        for conv in self.model.convs:
            Hs.append(conv.H.data.clone().cpu().numpy())
        return np.array(Hs)