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
    def __init__(self, is_pre_trained, feature_extract=cfg.FEATURE_EXTRACT, output_dim=cfg.NUM_CLASSES):
        super(HowDoIFeel, self).__init__()
        self.model = None
        self.input_size = 0
        self.is_pre_trained = is_pre_trained
        weights = 'DEFAULT' if cfg.MODEL == cfg.PERSONAL else None
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
        elif cfg.MODEL == cfg.PERSONAL:
            self.features = self._make_layers(cfg.NUM_INPUT_CHANNELS)
            self.classifier = nn.Sequential(nn.Linear(6 * 6 * 128, 64),
                                            nn.ELU(True),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(64, output_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.is_pre_trained:
            return self.model(x)
        else:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = nn.functional.dropout(out, p=cfg.DROPOUT, training=True)
            out = self.classifier(out)
            return out

    @staticmethod
    def _make_layers(in_channels):
        layers = []
        for x in cfg.NETWORK_CONFIG:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)




