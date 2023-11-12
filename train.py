import torch
import torch.nn as nn
from dataclasses import dataclass
import generate_dataset as gd
from early_stopping import Early_Stopping


@dataclass
class TrainingConfig:
    epochs :int
    val_num_batches :int
    batch_size :int
    lr :float
    device: torch.device
    A_size :int
    optimizer :torch.optim.Optimizer
    early_stopping :Early_Stopping
    criterion: nn.Module = nn.MSELoss()

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
        self.optimizer = config.optimizer(model.parameters(), lr=config.lr)
        self.criterion = config.criterion
        self.A_size = config.A_size

        # setup early stopping
        self.early_stopping = config.early_stopping

        # train stats
        self.train_loss = [ 0 for _ in range(self.epochs) ]
        self.val_loss = [ 0 for _ in range(self.epochs) ]

    def train(self ) -> nn.Module:

        for i in range(self.epochs):
            self.train_loss[i] = 0   # <--- initialize epoch loss

            X, y = self.dataset.gen_batch(self.batch_size, self.A_size) # <--- generate batch
            X, y = X.to(self.device), y.to(self.device)                 # <--- move to GPU

            y_hat = self.model(X)
        
            # check for shape mishaps before loss calculation
            assert y_hat.shape == y.shape, f"y_hat shape: {y_hat.shape} | y shape: {y.shape}"
            loss = self.criterion(y_hat, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.train_loss[i] = loss.item() / self.batch_size # average epooch loss over batch
            self.validate(i) # <--- validate on epoch

            # check for early stopping 
            self.early_stopping(self.val_loss[i], self.model)
            if self.early_stopping.stop_training:
                self.model.load_state_dict(self.early_stopping.best_model) # set model to best historical model
                break

        return self.model



    def validate(self, epoch :int) -> nn.Module:
        
        for i in range(self.val_num_batches):

            X, y = self.dataset.gen_batch(self.batch_size, self.A_size) # <--- generate batch
            X, y = X.to(self.device), y.to(self.device)       # <--- move to GPU
            
            with torch.no_grad():
                y_hat = self.model(X)
                loss = self.criterion(y_hat, y)

            self.val_loss[epoch] += loss.item() / self.batch_size # average epoch loss over batch
        self.val_loss[epoch] /= self.val_num_batches # average epoch loss num validation batchs 