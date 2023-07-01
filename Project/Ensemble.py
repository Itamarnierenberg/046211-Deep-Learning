import Config as cfg
from HowIFeel import HowIFeel
import torch
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tools import plot_confusion_matrix
import numpy as np


if not cfg.ENABLE_ENSEMBLE:
    raise ValueError("Ensamble Disabled in Config.py")
device = cfg.GPU_STR if torch.has_mps else "cpu"
is_pre_trained = False if cfg.MODEL == cfg.PERSONAL_1 or cfg.MODEL == cfg.PERSONAL_2 or cfg.PERSONAL_3 or cfg.PERSONAL_VGG else True

# Define the test transform for evaluation
test_transform = transforms.Compose([
    Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
    Grayscale(num_output_channels=cfg.NUM_INPUT_CHANNELS),
    ToTensor()
])
test_data = datasets.ImageFolder(cfg.TEST_DIRECTORY, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# Load the models that are defined in the Config.py file
models_list = list()
for model_params in cfg.ENSEMBLE_MODELS:
    print(model_params)
    curr_model = HowIFeel(is_pre_trained, model_name=model_params[0])
    curr_model_weights = torch.load(model_params[1])
    curr_model.load_state_dict(curr_model_weights)
    curr_model.to(device)
    curr_model.eval()
    models_list.append(curr_model)

with torch.set_grad_enabled(False):

    # initialize a list to keep track of our predictions
    predictions = []

    # iterate through the test set
    for (data, _) in test_loader:
        # move the data into the device used for testing
        data = data.to(device)
        outputs = list()
        for model in models_list:
            # perform a forward pass and calculate the training loss
            curr_output = model(data)
            output = model(data)
            outputs.append(curr_output.argmax(axis=1).cpu().numpy())
        ensemble_output = np.zeros(len(data))
        for i in range(len(outputs[0])):
            count_list = np.zeros(cfg.NUM_CLASSES)
            for j in range(len(models_list)):
                count_list[outputs[j][i]] += 1
            ensemble_output[i] = np.argmax(count_list)
        predictions.extend(ensemble_output)

# evaluate the network
print("[INFO] evaluating network...")
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))
con_mat = confusion_matrix(actual, predictions)
plot_confusion_matrix(con_mat, test_data.classes)
test_acc = (predictions == np.array(actual)).sum() / len(predictions)
print(f"[INFO] Final Test Accuracy = {test_acc}")
