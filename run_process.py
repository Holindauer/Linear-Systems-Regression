import torch
import argparse
from train import Trainer
from train import TrainingConfig
from simple_model import Model
from generate_dataset import Dataset
from early_stopping import Early_Stopping
import sys

class Training_Args:

    def __init__(self):
        # Initialize the argument parser
        self.parser = argparse.ArgumentParser(description='Training Arguments')
        self._add_arguments()
        self.args = self.parser.parse_args()

    def _add_arguments(self):
        # Arguments for Early_Stopping
        self.parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
        self.parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum delta for early stopping')

        # Arguments for TrainingConfig
        self.parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
        self.parser.add_argument('--val_num_batches', type=int, default=10, help='Number of validation batches')
        self.parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        self.parser.add_argument('--A_size', type=int, default=4, help='Size of A')

        # Arguments for Dataset
        self.parser.add_argument('--matrix_type', type=str, default='sparse', help='Matrix type')
        self.parser.add_argument('--fill_percentage', type=float, default=0.1, help='Fill percentage')

    @staticmethod
    def run_process():
        args_obj = Training_Args()

        print("Training Model...")

        # setup early stopping class
        early_stopping = Early_Stopping(
            patience=args_obj.args.patience,
            min_delta=args_obj.args.min_delta
        )

        # setup training config
        config = TrainingConfig(
            epochs=args_obj.args.epochs,
            val_num_batches=args_obj.args.val_num_batches,
            batch_size=args_obj.args.batch_size,
            lr=args_obj.args.lr,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            optimizer=torch.optim.Adam,
            criterion=torch.nn.MSELoss(),
            A_size=args_obj.args.A_size,
            early_stopping=early_stopping
        )
        
        # setup dataset
        dataset = Dataset(
            matrix_type=args_obj.args.matrix_type,
            fill_percentage=args_obj.args.fill_percentage
        )

        # setup model
        model = Model(config.A_size)

        # setup trainer
        trainer = Trainer(dataset, config, model)

        # run training
        trained_model = trainer.train()


# This is to ensure that the script runs only when it is executed directly, not when imported
if __name__ == "__main__":
    Training_Args.run_process()
