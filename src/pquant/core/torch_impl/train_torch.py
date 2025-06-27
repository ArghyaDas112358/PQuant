import torch
from tqdm.auto import tqdm  
import collections

from pquant.core.torch_impl.compressed_layers_torch import (
    call_post_round_functions,
    post_epoch_functions,
    post_pretrain_functions,
    pre_epoch_functions,
    pre_finetune_functions,
    save_weights_functions,
)

class History:
    def __init__(self):
        self.history = collections.defaultdict(list)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history[k].append(v)

def iterative_train_torch(model, config, train_func, valid_func, **kwargs):
    """
    Generic training loop, user provides training and validation functions
    """
    epoch = torch.tensor(0)  # Keeps track of all the epochs completed
    training_config = config["training_parameters"]
    print("Ypp")

    # Pre-training loop 
    if training_config["pretraining_epochs"] > 0:
        pretrain_epochs = range(training_config["pretraining_epochs"])
        pbar = tqdm(pretrain_epochs, desc="Pre-training")
        for e in pbar:
            model.train()
            pre_epoch_functions(model, e, training_config["pretraining_epochs"])
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["pretraining_epochs"])
            epoch += 1
        post_pretrain_functions(model, config)

    # Main training loop 
    rounds = range(training_config["rounds"])
    outer_pbar = tqdm(rounds, desc="Training Rounds")
    for r in outer_pbar:
        
        outer_pbar.set_description(f"Round {r + 1}/{training_config['rounds']}")
        inner_epochs = range(training_config["epochs"])
        inner_pbar = tqdm(inner_epochs, desc="Epochs", leave=False)
        
        for e in inner_pbar:
            model.train()
            if r == 0 and training_config["save_weights_epoch"] == e:
                save_weights_functions(model)
            pre_epoch_functions(model, e, training_config["epochs"])
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["epochs"])
            epoch += 1
        call_post_round_functions(model, training_config["rewind"], training_config["rounds"], r)

    # Fine-tuning loop
    pre_finetune_functions(model)
    if training_config["fine_tuning_epochs"] > 0:
        finetune_epochs = range(training_config["fine_tuning_epochs"])
        pbar = tqdm(finetune_epochs, desc="Fine-tuning")
        for e in pbar:
            model.train()
            pre_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            train_func(model, epoch=epoch, **kwargs)
            model.eval()
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config["fine_tuning_epochs"])
            epoch += 1
            
    return model