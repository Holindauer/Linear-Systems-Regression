import torch

from train import Trainer
from train import TrainingConfig
from model import Model
from generate_dataset import Dataset


if __name__ == "__main__":

    print("Starting...")

    # setup training config
    config = TrainingConfig(
        epochs=1000,
        val_num_batches=10,
        batch_size=100,
        lr=0.001,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss(),
        A_size=3
       )

    # setup dataset
    dataset = Dataset()

    # setup model
    model = Model(config.A_size)

    # setup trainer
    trainer = Trainer(dataset, config, model)

    # run trianing
    trainer.train()






