import torch.nn as nn
import Config as cfg
from torchvision.models import resnet18
from torchvision.models import vgg13


def set_parameter_requires_grad(model, feature_extracting=cfg.FEATURE_EXTRACT):
    if feature_extracting:
        model.requires_grad_(False)
    else:  # fine-tuning
        model.requires_grad_(True)


class HowDoIFeel(nn.Module):
    def __init__(self, feature_extract=cfg.FEATURE_EXTRACT, output_dim=cfg.NUM_CLASSES):
        super(HowDoIFeel, self).__init__()
        self.model = None
        self.input_size = 0
        self.is_pre_trained = False
        weights = None if cfg.MODEL == cfg.PERSONAL else None
        if cfg.MODEL == cfg.RES_NET_18:
            """ Resnet18 """
            self.model = resnet18(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_dim)  # replace the last FC layer
            self.input_size = 224
            self.is_pre_trained = True
        elif cfg.MODEL == cfg.VGG_13:
            self.model = vgg13(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, output_dim)
            self.input_size = 224
            self.is_pre_trained = True
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.is_pre_trained:
            return self.model(x)
        else:
            raise NotImplementedError




