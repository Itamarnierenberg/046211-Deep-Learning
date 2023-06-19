from time import time
from copy import deepcopy
from tqdm import tqdm
import Config as cfg
import torch
from tools import LRScheduler
from tools import EarlyStopping


device = "mps" if torch.has_mps else "cpu"
def train_model(model, dataloaders, criterion, optimizer, num_epochs=cfg.NUM_OF_EPOCHS):
    if cfg.ENABLE_SCHEDULER:
        lr_scheduler = LRScheduler(optimizer)
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
                        optimizer.step()
                        train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                        total_train_loss += loss
                    else:
                        total_val_loss += loss
                        val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

                # statistics
            if phase == 'train':
                avg_epoch_loss = total_train_loss / train_steps
                train_correct = train_correct / len(dataloaders[phase].dataset)
                history['train_loss'].append(avg_epoch_loss.cpu().detach().numpy())
                history['train_acc'].append(train_correct)
                print('{} Loss: {:.4f} Acc: {:.4f} Learning Rate: {:.4f}'.format(phase, avg_epoch_loss, train_correct, cfg.LR))
            else:
                avg_epoch_loss = total_val_loss / val_steps
                val_correct = val_correct / len(dataloaders[phase].dataset)
                history['val_loss'].append(avg_epoch_loss.cpu().detach().numpy())
                history['val_acc'].append(val_correct)
                print('{} Loss: {:.4f} Acc: {:.4f} Learning Rate: {:.4f}'.format(phase, avg_epoch_loss, val_correct, cfg.LR))
                validation_loss = avg_epoch_loss.cpu().detach().numpy()
                if cfg.ENABLE_SCHEDULER:
                    lr_scheduler(validation_loss)
                if cfg.ENABLE_EARLY_STOPPING:
                    early_stopping(validation_loss)

            # deep copy the model
            if phase == 'val' and val_correct > best_acc:
                best_acc = val_correct
                best_model_wts = deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(val_correct)
        print()
        if cfg.ENABLE_EARLY_STOPPING and early_stopping.early_stop_enabled:
            break

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, history
