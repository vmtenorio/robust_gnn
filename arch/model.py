
import torch.nn as nn
import torch

import numpy as np
from arch.arch import GCNModel1, GCNModel2

from opt import graph_id, graph_id_rew

class Model:
    def __init__(self, model, n_epochs, lr, wd, eval_freq):
        self.model = model

        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd

        self.eval_freq = eval_freq

    def predict(self, x):
        return self.model(x)

    def test_reg(self, S, x_train, y_train, gamma, x_test=None, y_test=None, verbose=True):

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        loss_fn = nn.MSELoss()

        loss_train, loss_test = [np.zeros(self.n_epochs) for _ in range(2)]

        for i in range(self.n_epochs):
            y = self.model(x_train)
            loss = loss_fn(y, y_train)

            assert y.shape[0] == S.shape[0]

            # Commutativity term
            #loss += gamma*((model.H @ S - S @ model.H)**2).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_train[i] = loss.cpu().item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose and (x_test is not None):
                with torch.no_grad():
                    y_hat = self.model(x_test).T
                    loss_test = loss_fn(y_hat, y_test.T)
                print(f"Epoch {i+1}/{self.n_epochs} - Loss train: {loss.cpu().item()} - Loss: {loss_test.cpu().item()}", flush=True)
            elif (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_epochs} - Loss train: {loss.cpu().item()}", flush=True)

        return loss_train, loss_test

    def test_clas(self, x, labels, train_idx=[], val_idx=[], test_idx=[], verbose=True):

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        loss_fn = nn.CrossEntropyLoss()

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_epochs) for _ in range(4)]

        for i in range(self.n_epochs):
            y = self.model(x)
            loss = loss_fn(y[train_idx], labels[train_idx])

            # Commutativity term
            #loss += gamma*((model.H @ S - S @ model.H)**2).sum()

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

    def test_iterative(self, Sn, x, labels, lambd, gamma, n_iters, train_idx=[], val_idx=[], test_idx=[]):

        S_id = Sn

        accs_train = np.zeros((n_iters, self.n_epochs))
        accs_test = np.zeros((n_iters, self.n_epochs))

        for i in range(n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            S_gcn = torch.Tensor(S_id).to(x.device)

            self.model.update_Spow(S_gcn)
            
            # Filter estimation
            accs_train[i,:], _, accs_test[i,:], _ = self.test_clas(S_gcn, x, labels, gamma, train_idx, val_idx, test_idx, verbose=False)

            h_id = self.model.h.data
            H_id = torch.sum(h_id[:,None,None]*self.model.Spow, 0).cpu().numpy()

            #print("Graph identification")
            # Graph estimation
            S_id = graph_id(Sn, H_id, np.zeros(Sn.shape), lambd, gamma, 0.)

        return accs_train, accs_test, S_id
    
    def eval_acc(self, x, labels):
        y_hat = self.predict(x)

        preds = torch.argmax(y_hat, 1)
        results = (preds == labels).type(torch.float32)
        return results.mean().item()
    

# TODO: Possible Merge of the two models?
# TODO: if not, common methods for parent class?
###############################################################
######################## MODEL 1 ##############################
###############################################################

class Model1:
    def __init__(self, S, n_epochs, lr, wd, eval_freq, model_params, n_iters):

        self.model = GCNModel1(S=S, **model_params)

        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd

        self.eval_freq = eval_freq

        self.n_iters = n_iters

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def stepH(self, x, labels, train_idx, val_idx, test_idx, verbose=False):

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_epochs) for _ in range(4)]

        for i in range(self.n_epochs):
            self.model.train()
            y = self.model(x)
            loss = self.loss_fn(y[train_idx], labels[train_idx])

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Compute accuracy on training/validation/test
            acc_train[i] = self.evaluate(x, labels, train_idx)
            acc_val[i] = self.evaluate(x, labels, val_idx)
            acc_test[i] = self.evaluate(x, labels, test_idx)
            losses[i] = loss.item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_epochs} - Loss: {loss.item()} - Train Acc: {acc_train[i]} - Test Acc: {acc_test[i]}", flush=True)

        return acc_train, acc_val, acc_test, losses
    
    def stepS(self, S, Sn, lambd, beta):
        # Gradient step?

        # Proximal for Sparsity
        S = torch.sign(S) * torch.maximum(torch.abs(S) - beta, torch.zeros(S.shape, device=S.device))

        # Proximal for distance to S_bar
        idxs_greater = torch.where(S - Sn > lambd)
        idxs_lower = torch.where(S - Sn < -lambd)
        S_prox = torch.zeros(S.shape, device=S.device)
        S_prox[idxs_greater] = S[idxs_greater] - lambd
        S_prox[idxs_lower] = S[idxs_lower] + lambd
        S = S_prox

        # Projection onto \mathcal{S}
        S = torch.where(S < 0., 0., S)
        S = torch.where(S > 1., 1., S)
        S = (S + S.T) / 2

        return S

    
    def test_model(self, Sn, x, labels, lambd, beta, merge_theta_H=False, train_idx=[], val_idx=[], test_idx=[], verbose=False):
        S_id = Sn

        accs_train = np.zeros((self.n_iters, self.n_epochs))
        accs_test = np.zeros((self.n_iters, self.n_epochs))

        norm_S = torch.linalg.norm(Sn)

        if not merge_theta_H:
            self.model.deactivate_h()

        for i in range(self.n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            if not merge_theta_H:
                self.model.swap_params()
                # Theta estimation
                accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx, test_idx, verbose=verbose)
                self.model.swap_params()
            
            # Filter estimation
            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(x, labels, train_idx, val_idx, test_idx, verbose=verbose)

            #print("Graph identification")
            # Graph estimation
            S_id = self.stepS(S_id, Sn, lambd, beta)

            self.model.update_Spow(S_id)

            if verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err Sn: {torch.linalg.norm(S_id - Sn) / norm_S}")

        return accs_train, accs_test, S_id
    
###############################################################
######################## MODEL 2 ##############################
###############################################################

class Model2:
    def __init__(self, S, n_epochs, lr, wd, eval_freq, model_params, n_iters):

        self.model = GCNModel2(S=S, **model_params)

        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd

        self.eval_freq = eval_freq

        self.n_iters = n_iters

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)
    
    def stepH(self, S, x, labels, gamma, train_idx, val_idx, test_idx, verbose=False):

        acc_train, acc_val, acc_test, losses = [np.zeros(self.n_epochs) for _ in range(4)]

        for i in range(self.n_epochs):
            self.model.train()
            y = self.model(x)
            loss = self.loss_fn(y[train_idx], labels[train_idx])

            # Commutativity term
            loss += gamma*self.model.commutativity_term(S)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Compute accuracy on training/validation/test
            acc_train[i] = self.evaluate(x, labels, train_idx)
            acc_val[i] = self.evaluate(x, labels, val_idx)
            acc_test[i] = self.evaluate(x, labels, test_idx)
            losses[i] = loss.item()

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"Epoch {i+1}/{self.n_epochs} - Loss: {loss.item()} - Train Acc: {acc_train[i]} - Test Acc: {acc_test[i]}", flush=True)

        return acc_train, acc_val, acc_test, losses

    def stepS(self, S, Sn, gamma, lambd, beta):
        # Gradient step
        for layer in self.model.convs:
            H = layer.H.data
            S -= 2*gamma*(H.T @ H @ S - H.T @ S @ H - H @ S @ H.T + S @ H @ H.T)

        # Proximal for Sparsity
        S = torch.sign(S) * torch.maximum(torch.abs(S) - beta, torch.zeros(S.shape, device=S.device))

        # Proximal for distance to S_bar
        idxs_greater = torch.where(S - Sn > lambd)
        idxs_lower = torch.where(S - Sn < -lambd)
        S_prox = torch.zeros(S.shape, device=S.device)
        S_prox[idxs_greater] = S[idxs_greater] - lambd
        S_prox[idxs_lower] = S[idxs_lower] + lambd
        S = S_prox

        # Projection onto \mathcal{S}
        S = torch.where(S < 0., 0., S)
        S = torch.where(S > 1., 1., S)
        S = (S + S.T) / 2

        return S
        

    def test_model(self, Sn, x, labels, lambd, gamma, beta, merge_theta_H=False, train_idx=[], val_idx=[], test_idx=[], verbose=False):
        S_id = Sn

        accs_train = np.zeros((self.n_iters, self.n_epochs))
        accs_test = np.zeros((self.n_iters, self.n_epochs))

        norm_S = torch.linalg.norm(Sn)

        if not merge_theta_H:
            self.model.deactivate_H()

        for i in range(self.n_iters):
            #print("**************************************")
            #print(f"************ Iteration {i} ***********", end="")
            #print("**************************************")

            if not merge_theta_H:
                self.model.swap_params()
                # Theta estimation
                accs_train[i,:], _, accs_test[i,:], _ = self.stepH(S_id, x, labels, gamma, train_idx, val_idx, test_idx, verbose=verbose)
                self.model.swap_params()
            
            # Filter estimation
            accs_train[i,:], _, accs_test[i,:], _ = self.stepH(S_id, x, labels, gamma, train_idx, val_idx, test_idx, verbose=verbose)

            #print("Graph identification")
            # Graph estimation
            S_id = self.stepS(S_id, Sn, gamma, lambd, beta)

            if verbose:
                print(f"Iteration {i+1} DONE - Acc Test: {accs_test[i,-1]} - Err Sn: {torch.linalg.norm(S_id - Sn) / norm_S}")

        return accs_train, accs_test, S_id


