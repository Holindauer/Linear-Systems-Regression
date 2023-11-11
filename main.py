import torch

from train import Trainer
from train import TrainingConfig
from model import Model
from generate_dataset import Dataset
from early_stopping import Early_Stopping


if __name__ == "__main__":

    print("Starting...")

    # setup early stopping class
    early_stopping = Early_Stopping(
        patience=10,
        min_delta=0.001
    )

    # setup training config
    config = TrainingConfig(
        epochs=10000,
        val_num_batches=10,
        batch_size=100,
        lr=0.001,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss(),
        A_size=3,
        early_stopping=early_stopping
       )
    
    # setup dataset
    dataset = Dataset()

    # setup model
    model = Model(config.A_size)

    # setup trainer
    trainer = Trainer(dataset, config, model)

    # run trianing
    trainer.train()






