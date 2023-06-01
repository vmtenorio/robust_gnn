
import torch.nn as nn
import torch

import numpy as np
from arch.arch import GCNModel1, GCNModel2

from sev_filters_opt import graph_id


## Non-convex Robust GNN Models

class RobustGNNModel:
    def __init__(self, S, n_epochs, lr, wd, lr_S, eval_freq, model_params, n_iters_out, n_iters_S, problem_type="clas"):

        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_S = lr_S
        self.wd = wd

        self.eval_freq = eval_freq

        self.n_iters_out = n_iters_out
        self.n_iters_S = n_iters_S

        self.build_model(S.clone(), model_params)
        
        self.problem_type = problem_type

        if self.problem_type == "clas":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.problem_type == "reg":
            self.loss_fn = nn.MSELoss()

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

    def stepH(self, x, labels, train_idx, val_idx, test_idx, S=None, verbose=False):

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_epochs) for _ in range(4)]

        for i in range(self.n_epochs):
            self.model.train()
            y = self.model(x).squeeze()
            loss = self.calc_loss(y[train_idx], labels[train_idx])

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
                print(f"Epoch {i+1}/{self.n_epochs} - Loss: {loss.item()} - Train Acc: {acc_train[i]} - Test Acc: {acc_test[i]}")

        return acc_train, acc_val, acc_test, losses
    
    def stepS(self, Sn, x, labels, gamma, lambd, beta, train_idx, S_true=None, norm_S=True):

        errs_S = np.zeros(self.n_iters_S)
        change_S = np.zeros(self.n_iters_S)
        norm_S = torch.linalg.norm(S_true)

        for i in range(self.n_iters_S):
            orig_S = self.model.S.data.clone()

            S = self.model.S.data.clone()

            S = self.gradient_step_S(S, gamma, x, labels, train_idx)

            # Proximal for Sparsity
            #S = torch.sign(S) * torch.maximum(torch.abs(S) - beta, torch.zeros(S.shape, device=S.device))

            # Proximal for the distance to S_bar
            idxs_greater = torch.where(S - Sn > lambd)
            idxs_lower = torch.where(S - Sn < -lambd)
            S_prox = Sn.clone()
            S_prox[idxs_greater] = S[idxs_greater] - lambd
            S_prox[idxs_lower] = S[idxs_lower] + lambd
            S = S_prox

            # Projection onto \mathcal{S}
            S = torch.clamp(S, min=0., max=1.)
            S = (S + S.T) / 2

            #errs_S[i] = torch.linalg.norm(S - S_true) / norm_S
            
            errs_S[i] = torch.linalg.norm(S/torch.linalg.norm(S) - S_true/norm_S)
            change_S[i] = torch.linalg.norm(S - orig_S) / norm_S

            self.model.update_S(S, normalize=norm_S)
            
        return errs_S, change_S

    
    def test_model(self, Sn, x, labels, gamma, lambd, beta, train_idx=[], val_idx=[], test_idx=[], norm_S=True, S_true=None, verbose=False):

        accs_train = np.zeros((self.n_iters_out, self.n_epochs))
        accs_test = np.zeros((self.n_iters_out, self.n_epochs))
        errs_S = np.zeros((self.n_iters_out, self.n_iters_S))
        change_S = np.zeros((self.n_iters_out, self.n_iters_S))

        for i in range(self.n_iters_out):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")
            errs_S[i,:], change_S[i,:] = self.stepS(Sn, x, labels, gamma, lambd, beta, train_idx, S_true, norm_S)

            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx, test_idx, )

            #print("Graph identification")
            # Graph estimation

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err S: {errs_S[i,-1]}")

        return accs_train, accs_test, self.model.S.data, errs_S, change_S
    

###############################################################
######################## MODEL 1 ##############################
###############################################################
class RobustGNNModel1(RobustGNNModel):

    def build_model(self, S, model_params):
        self.model = GCNModel1(S=S, **model_params)
        
        self.opt_hW = torch.optim.Adam(
            [layer.h for layer in self.model.convs] +
            [layer.W for layer in self.model.convs] +
            [layer.b for layer in self.model.convs],
            lr=self.lr, weight_decay=self.wd)
        
        self.opt_S = torch.optim.SGD(
            [self.model.S],
            lr=self.lr_S)

    def calc_loss(self, y_hat, y_train):
        return self.loss_fn(y_hat, y_train)
    
    def gradient_step_S(self, S=None, gamma=None, x=None, y=None, train_idx=None):
        y_hat = self.model(x).squeeze()
        loss = self.calc_loss(y_hat[train_idx], y[train_idx])

        self.opt_S.zero_grad()
        loss.backward()
        self.opt_S.step()

        return self.model.S.data
        
###############################################################
######################## MODEL 2 ##############################
###############################################################

class RobustGNNModel2(RobustGNNModel):

    def build_model(self, S, model_params):
        self.model = GCNModel2(S=S, **model_params)

        self.opt_hW = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def calc_loss(self, y_hat, y_train):
        return self.loss_fn(y_hat, y_train) + self.model.commutativity_term()
    
    def gradient_step_S(self, S, gamma, x=None, y=None, train_idx=None):
        for layer in self.model.convs:
            H = layer.H.data
            S -= 2*gamma*(H.T @ H @ S - H.T @ S @ H - H @ S @ H.T + S @ H @ H.T)
        return S
    




###############################################################
############## Convex alternatives to estimate S ##############
###############################################################
class ModelCVX:
    def __init__(self, S, n_epochs, lr, wd, eval_freq, model_params):

        self.n_epochs = n_epochs
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

        loss_train, loss_test = [np.zeros(self.n_epochs) for _ in range(2)]

        for i in range(self.n_epochs):
            self.model.train()
            y_hat = self.model(X).squeeze()
            loss = self.calc_loss(y_hat[train_idx], Y[train_idx], S)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_train[i] = loss.cpu().item()
            loss_test[i] = self.evaluate_reg(X, Y, test_idx)

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_epochs} - Loss train: {loss_train[i]} - Loss: {loss_test[i]}", flush=True)

        return loss_train, loss_test

    def test_clas(self, x, labels, train_idx=[], val_idx=[], test_idx=[], S=None, verbose=True):

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        loss_fn = nn.CrossEntropyLoss()

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_epochs) for _ in range(4)]

        for i in range(self.n_epochs):
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
                print(f"Epoch {i+1}/{self.n_epochs} - Loss: {loss.item()} - Train Acc: {acc_train[i]} - Test Acc: {acc_test[i]}", flush=True)

        return acc_train, acc_val, acc_test, losses

    def test_iterative_clas(self, Sn, x, labels, lambd, gamma, n_iters, train_idx=[], val_idx=[], test_idx=[]):

        S_id = Sn

        accs_train = np.zeros((n_iters, self.n_epochs))
        accs_test = np.zeros((n_iters, self.n_epochs))

        for i in range(n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            S_gcn = torch.Tensor(S_id).to(x.device)

            self.model.update_S(S_gcn)
            
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

        loss_train = np.zeros((n_iters, self.n_epochs))
        loss_test = np.zeros((n_iters, self.n_epochs))
        errs_S = np.zeros((n_iters))

        norm_S = np.linalg.norm(S_true)

        for i in range(n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            S_gcn = torch.Tensor(S_id).to(x.device)

            change_S = np.linalg.norm(self.model.S.data.cpu().numpy() - S_id)

            self.model.update_S(S_gcn)
            
            # Filter estimation
            loss_train[i,:], loss_test[i,:] = self.test_reg(S_gcn, x, labels, train_idx, val_idx, test_idx, verbose=False)

            Hs = self.build_filters()

            #print("Graph identification")
            # Graph estimation
            S_id = graph_id(Sn.cpu().numpy(), Hs, np.zeros(Hs.shape) if Cy is None else Cy, lambd, gamma, 10.)

            errs_S[i] = np.linalg.norm(S_id - S_true) / norm_S

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
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
        Spow = torch.stack([torch.linalg.matrix_power(self.model.S.data, k) for k in range(self.model.K)])
        for conv in self.model.convs:
            h_id = conv.h.data
            Hs.append(torch.sum(h_id[:,None,None]*Spow, 0).cpu().numpy())
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