import torch.nn as nn
import Config as cfg
from torchvision.models import resnet18
from torchvision.models import vgg13
from transformers import ViTModel, ViTConfig


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
        weights = 'DEFAULT' if is_pre_trained else None
        if cfg.MODEL == cfg.RES_NET_18:
            """ Resnet18 """
            self.model = resnet18(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=self.model.conv1.bias)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_dim, bias=False)  # replace the last FC layer
            self.input_size = 224
            self.is_pre_trained = True
        elif cfg.MODEL == cfg.VGG_13:
            self.model = vgg13(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=self.model.features[0].kernel_size, stride=self.model.features[0].stride, padding=self.model.features[0].padding, bias=self.features[0].conv1.bias)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, output_dim, bias=False)
            self.input_size = 224
            self.is_pre_trained = True
        elif cfg.MODEL == cfg.VIT:
            model_checkpoint = 'google/vit-base-patch16-224-in21k'
            self.model = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
            set_parameter_requires_grad(self.model, feature_extract)
            config = ViTConfig()
            self.classifier = (
                nn.Linear(config.hidden_size, output_dim)
            )
            self.is_pre_trained = True
        elif cfg.MODEL == cfg.PERSONAL_1:
            self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.ELU(inplace=True),
                                          nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ELU(inplace=True),
                                          nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ELU(inplace=True),
                                          nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Dropout(p=cfg.DROPOUT),
                                            nn.Linear(6 * 6 * 128, 64),
                                            nn.ELU(True),
                                            nn.Dropout(p=cfg.DROPOUT),
                                            nn.Linear(64, output_dim))
        elif cfg.MODEL == cfg.PERSONAL_2:
            self.features = nn.Sequential(
                                          nn.Conv2d(cfg.NUM_INPUT_CHANNELS, 32, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 64, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Dropout(p=cfg.DROPOUT),
                                          nn.Conv2d(64, 64, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 64, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 128, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(128, 128, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 256, kernel_size=3),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Dropout(p=cfg.DROPOUT),
            )
            self.classifier = nn.Sequential(
                                            nn.Flatten(),
                                            nn.Linear(256, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.DROPOUT),
                                            nn.Linear(1024, cfg.NUM_CLASSES, bias=False)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.is_pre_trained:
            if cfg.MODEL == cfg.VIT:
                x = self.model(x)['last_hidden_state']
                return self.classifier(x[:, 0, :])
            else:
                return self.model(x)
        else:
            out = self.features(x)
            # out = out.view(out.size(0), -1)
            # out = nn.functional.dropout(out, p=cfg.DROPOUT, training=True)
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




