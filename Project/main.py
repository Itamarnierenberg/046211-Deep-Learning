import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from HowDoIFeel import HowDoIFeel
import torch.nn as nn
from TrainModel import train_model
import Config as cfg
from tools import show_examples


# configure the device to use for training the model, either gpu or cpu
device = cfg.GPU_STR if torch.has_mps else "cpu"
print(f"[INFO] Current training device: {device}")

# initialize a list of preprocessing steps to apply on each image during
# training/validation and testing
train_transform = transforms.Compose([
    Grayscale(num_output_channels=cfg.NUM_INPUT_CHANNELS),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])

test_transform = transforms.Compose([
    Grayscale(num_output_channels=cfg.NUM_INPUT_CHANNELS),
    ToTensor()
])

# load all the images within the specified folder and apply different augmentation
train_data = datasets.ImageFolder(cfg.TRAIN_DIRECTORY, transform=train_transform)
test_data = datasets.ImageFolder(cfg.TEST_DIRECTORY, transform=test_transform)

classes = train_data.classes
print(f"[INFO] Class labels: {classes}")
# use train samples to generate train/validation set
num_train_samples = len(train_data)
train_size = int(np.floor(num_train_samples * cfg.TRAIN_SIZE))
val_size = int(np.ceil(num_train_samples * cfg.VAL_SIZE))

print(f'[INFO] Number of Training Samples = {train_size}')
print(f'[INFO] Number of Validation Samples = {val_size}')
print(f'[INFO] Number of total Samples = {num_train_samples}')
assert train_size + val_size == num_train_samples

# randomly split the training dataset into train and validation set
train_data, val_data = random_split(train_data, [train_size, val_size])

# modify the data transform applied towards the validation set
val_data.dataset.transforms = test_transform

# get the labels within the training set
train_classes = [label for _, label in train_data]

# count each labels within each classes
class_count_train = Counter(train_classes)
print(f"[INFO] Total sample: {class_count_train}")
print(f'[INFO] Train Data Summarize:')
for label, num_samples in class_count_train.items():
    print(f'\t[INFO] Emotion: {cfg.CODE_TO_STR[label]}, Samples: {num_samples}')

train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE)    # FIXME - Maybe add sampler=sampler
val_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# Visualize a few images from the data
show_examples(train_data)

pre_trained_model = HowDoIFeel()
pre_trained_model = pre_trained_model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  fine-tuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = pre_trained_model.parameters()
print(params_to_update)
print("[INFO] Params to learn:")
if cfg.FEATURE_EXTRACT:
    params_to_update = []  # override the initial list definition above
    for name, param in pre_trained_model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t[INFO] {name}")
else:
    for name, param in pre_trained_model.named_parameters():
        if param.requires_grad:
            print(f"\t[INFO] {name}")

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(params_to_update, lr=cfg.LR)
data_loaders = {'train': train_loader, 'val': val_loader}
criterion = nn.CrossEntropyLoss()

model, _, history = train_model(pre_trained_model, data_loaders, criterion, optimizer)
# move model back to cpu and save the trained model to disk
if device == cfg.GPU_STR:
    model = model.to(cfg.CPU_STR)
torch.save(model.state_dict(), f'{cfg.RESULTS_DIRECTORY}/{cfg.MODEL_FILE}')
# plot the training loss and accuracy overtime
plt.style.use("ggplot")
plt.figure()
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylabel('Loss/Accuracy')
plt.xlabel("#No of Epochs")
plt.title('Training Loss and Accuracy on FER2013')
plt.legend(loc='upper right')
plt.savefig(f'{cfg.RESULTS_DIRECTORY}/{cfg.PLOT_FILE}')

# evaluate the model based on the test set
model = model.to(device)
with torch.set_grad_enabled(False):
    # set the evaluation mode
    model.eval()

    # initialize a list to keep track of our predictions
    predictions = []

    # iterate through the test set
    for (data, _) in test_loader:
        # move the data into the device used for testing
        data = data.to(device)

        # perform a forward pass and calculate the training loss
        output = model(data)
        output = output.argmax(axis=1).cpu().numpy()
        predictions.extend(output)

# evaluate the network
print("[INFO] evaluating network...")
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))

