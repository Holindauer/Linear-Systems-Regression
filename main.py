import torch

from train import Trainer
from train import TrainingConfig
from model import Model
from generate_dataset import Dataset
from early_stopping import Early_Stopping
import numpy as np


if __name__ == "__main__":

    print("Starting...")

    # setup early stopping class
    early_stopping = Early_Stopping(
        patience=30,
        min_delta=0.0
    )

    # setup training config
    config = TrainingConfig(
        epochs=1000,
        val_num_batches=10,
        batch_size=100,
        lr=0.001,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss(),
        A_size=2,
        early_stopping=early_stopping
       )
    
    # setup dataset
    dataset = Dataset()

    # setup model
    model = Model(config.A_size)

    # setup trainer
    trainer = Trainer(dataset, config, model)

    # run trianing
    trained_model = trainer.train()

    # create a batch using the trained model and run a forward pass and print the targets next to the predictions
    X, y = dataset.gen_batch(10, config.A_size)
    y_hat = trained_model(X)
    print(f"\n\ny: {y.detach().cpu().numpy()}")
    print(f"y_hat: {y_hat.detach().cpu().numpy()}")






