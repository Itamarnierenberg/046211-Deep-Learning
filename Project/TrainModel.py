from time import time
from copy import deepcopy
from tqdm import tqdm
import Config as cfg
import torch
from tools import LRScheduler
from tools import EarlyStopping
from HowDoIFeel import define_model
import optuna
import torch.nn as nn


device = cfg.GPU_STR if torch.has_mps else "cpu"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model_or_trial, dataloaders, criterion, optimizer=None, num_epochs=cfg.NUM_OF_EPOCHS):
    if cfg.USE_OPTUNA:
        trial = model_or_trial
        model = define_model(trial).to(device)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        model = model_or_trial
        if cfg.ENABLE_SCHEDULER:
            if cfg.SCHED_NAME == 'ReduceLROnPlateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                                          patience=cfg.SCHED_PATIENCE,
                                                                          factor=cfg.REDUCE_FACTOR,
                                                                          verbose=True)
            elif cfg.SCHED_NAME == 'OneCycleLR':
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg.MAX_LEARNING_RATE,
                                                                   epochs=cfg.NUM_OF_EPOCHS, steps_per_epoch=len(dataloaders['train']))
        if cfg.ENABLE_EARLY_STOPPING:
            early_stopping = EarlyStopping()
    since = time()
    val_acc_history = []
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    train_steps = len(dataloaders['train'].dataset) // cfg.BATCH_SIZE
    val_steps = len(dataloaders['val'].dataset) // cfg.BATCH_SIZE

    # initialize a dictionary to save the training history
    history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": []
    }
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            total_train_loss = 0
            total_val_loss = 0
            train_correct = 0
            val_correct = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        if cfg.GRAD_CLIP:
                            nn.utils.clip_grad_value_(model.parameters(), cfg.GRAD_CLIP)
                        optimizer.step()
                        train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                        total_train_loss += loss
                        if cfg.ENABLE_SCHEDULER and cfg.SCHED_NAME == 'OneCycleLR':
                            lr_scheduler.step()
                    else:
                        total_val_loss += loss
                        val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

                # statistics
            if phase == 'train':
                avg_epoch_loss = total_train_loss / train_steps
                train_correct = train_correct / len(dataloaders[phase].dataset)
                history['train_loss'].append(avg_epoch_loss.cpu().detach().numpy())
                history['train_acc'].append(train_correct)
                print(f'\nLearning Rate = {get_lr(optimizer):f}, Training Loss = {avg_epoch_loss:.4f}, Training Accuracy = {train_correct:.4f}')
            else:
                avg_epoch_loss = total_val_loss / val_steps
                val_correct = val_correct / len(dataloaders[phase].dataset)
                history['val_loss'].append(avg_epoch_loss.cpu().detach().numpy())
                history['val_acc'].append(val_correct)
                print(f'\nLearning Rate = {get_lr(optimizer):f}, Validation Loss = {avg_epoch_loss:.4f}, Validation Accuracy = {val_correct:.4f}')
                validation_loss = avg_epoch_loss.cpu().detach().numpy()
                if cfg.USE_OPTUNA:
                    trial.report(val_correct, epoch)
                else:
                    if cfg.ENABLE_SCHEDULER and cfg.SCHED_NAME == 'ReduceLROnPlateau':
                        lr_scheduler.step(validation_loss)
                    if cfg.ENABLE_EARLY_STOPPING:
                        early_stopping(validation_loss)


            # deep copy the model
            if phase == 'val' and val_correct > best_acc:
                best_acc = val_correct
                best_model_wts = deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(val_correct)
        print()
        if not cfg.USE_OPTUNA and cfg.ENABLE_EARLY_STOPPING and early_stopping.early_stop_enabled:
            break
        if cfg.USE_OPTUNA and trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, history


