import torch
import torch.nn as nn

import numpy as np
import time

class Coach:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []

    def _train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        batch_loss, batch_correct = [], 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.model(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
            pred = torch.argmax(output, dim=1)
            batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def _test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        batch_loss, batch_correct = [], 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)

                batch_loss.append(loss.item())
                pred = torch.argmax(output, dim=1)
                batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def train_test(self):
        start = time.time()
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)

class CoachDML():
    def __init__(self, models, train_loader, test_loader, loss_fns, optimizers, device, epochs):
        self.model_1 = models[0]
        self.model_2 = models[1]

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_ce = loss_fns[0] # ce-loss
        self.loss_kl = loss_fns[1] # kl-loss

        self.optimizer_1 = optimizers[0]
        self.optimizer_2 = optimizers[1]

        self.device = device
        self.epochs = epochs

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []

    def _train_epoch(self):
        self.model_1.train()
        self.model_2.train()
        dataloader = self.train_loader
        batch_loss, batch_correct = [], 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output_1, output_2 = self.model_1(X), self.model_2(X)
            loss_1 = self.loss_ce(output_1, y) + self.loss_kl(output_1, output_2.detach())
            self.optimizer_1.zero_grad()
            loss_1.backward()
            self.optimizer_1.step()

            output_1 = self.model_1(X) # output again
            loss_2 = self.loss_ce(output_2, y) + self.loss_kl(output_2, output_1.detach())
            self.optimizer_2.zero_grad()
            loss_2.backward()
            self.optimizer_2.step()

            batch_loss.append(loss_1.item())
            pred = torch.argmax(output_1, dim=1)
            batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def _test_epoch(self):
        self.model_1.eval()
        self.model_2.eval()
        dataloader = self.test_loader
        batch_loss, batch_correct = [], 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output_1, output_2 = self.model_1(X), self.model_2(X)
                loss_1 = self.loss_ce(output_1, y) + self.loss_kl(output_1, output_2)

                batch_loss.append(loss_1.item())
                pred = torch.argmax(output_1, dim=1)
                batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def train_test(self):
        start = time.time()
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)