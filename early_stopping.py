import torch
import copy


"""
    This class implements early stopping to be called within the training loop. 
"""
class Early_Stopping:

    def __init__(self, patience :int = 10, min_delta :float = 0.0) -> None:

        # early stopping  hyper-params
        self.patience = patience
        self.min_delta = min_delta # <--- min loss differential considered an improvement
 
        # best stats
        self.best_loss = torch.inf
        self.best_epoch = 0
        self.best_model = None

        # current stats
        self.current_epoch = 0
        self.current_loss = None
        self.stopped_epoch = None
        self.stop_training = False

    def __call__(self, epoch_loss :float, model :torch.nn.Module) -> None:
        self.current_epoch += 1
        self.current_loss = epoch_loss

        if self.best_loss is None: # <--- init best stats
            self.best_loss = epoch_loss
            self.best_epoch = self.current_epoch
            self.best_model = model.state_dict() # <--- og model state dict 

        elif (epoch_loss - self.min_delta) < self.best_loss: # <--- check if loss improved
            self.best_loss = epoch_loss
            self.best_epoch = self.current_epoch
            self.best_model = copy.deepcopy(model.state_dict()) # <--- store model state dict if loss improved

        elif self.current_epoch - self.best_epoch > self.patience: # <--- check if patience exceeded
            self.stop_training = True
            self.stopped_epoch = self.current_epoch
            print(f"Early Stopping: Epoch {self.stopped_epoch} - loss: {self.current_loss:.4f} - best loss: {self.best_loss:.4f} - delta: {self.best_loss - self.current_loss:.4f}")