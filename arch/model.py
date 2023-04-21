
import torch.nn as nn
import torch

import numpy as np

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

    def test_clas(self, S, x, labels, gamma, train_idx=[], val_idx=[], test_idx=[], verbose=True):

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

