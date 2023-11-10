import torch
import torch.nn as nn
from dataclasses import dataclass
import generate_dataset as gd


@dataclass
class TrainingConfig:
    epochs :int
    val_num_batches :int
    batch_size :int
    lr :float
    device: torch.device
    optimizer :torch.optim.Optimizer
    criterion :nn.Module

class Trainer:
    def __init__(self, dataset :gd.Dataset, config :TrainingConfig, model :nn.Module) -> None:

        self.dataset = dataset
        self.model = model

        #unpack train config
        self.epochs = config.epochs
        self.val_num_batches = config.val_num_batches
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.device = config.device
        self.optimizer = config.optimizer
        self.criterion = config.criterion

        # train stats
        self.val_loss = []
        self.train_loss = []

    def train(self ) -> nn.Moduel:

        for i in range(self.epcohs):
            X, y = self.gen_batch(self.batch_size, self.size) # <--- generate batch
            X, y = X.to(self.device), y.to(self.device)       # <--- move to GPU

            for x, y in zip(X, y): # <--- train on batch
                y_hat = self.model(x)
                loss = nn.MSELoss(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.train_loss[i] += loss.item() # accumualte loss

            self.train_loss[i] /= self.batch_size # average epooch loss over batch
            self.validate(i) # <--- validate on epoch


            print(f"Epoch {i} | Train Loss: {self.train_loss[i]} | Val Loss: {self.val_loss[i]}")

        return self.model



    def validate(self, epoch :int) -> nn.Moduel:
        
        for i in range(self.val_num_batches):

            X, y = self.gen_batch(self.batch_size, self.size) # <--- generate batch
            X, y = X.to(self.device), y.to(self.device)       # <--- move to GPU
            
            with torch.no_grad():
                for x, y in zip(X, y):
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                    self.val_loss[epoch] += loss.item() # accumulate loss

            self.val_loss[i] /= self.batch_size # average epoch loss over batch
        self.val_loss[i] /= self.val_num_batches # average epoch loss num validation batchs 