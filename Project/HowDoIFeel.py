import torch
import torch.nn as nn
import Config as cfg
from torchvision.models import resnet18
from torchvision.models import vgg13
from transformers import ViTModel, ViTConfig
from facenet_pytorch import MTCNN, InceptionResnetV1


def set_parameter_requires_grad(model, feature_extracting=cfg.FEATURE_EXTRACT):
    """
    This function will set the require grad for each of the parameters.
    If feature_extracting is True, we will perform Transfer Learning, o.w we will fine tune the parameters
    :param model: The pytorch model
    :param feature_extracting: True for complete transfer learning, false for fine tuning
    :return: No return value
    """
    if feature_extracting:
        model.requires_grad_(False)
    else:  # fine-tuning
        model.requires_grad_(True)


class HowDoIFeel(nn.Module):
    def __init__(self, is_pre_trained, feature_extract=cfg.FEATURE_EXTRACT, output_dim=cfg.NUM_CLASSES, model_name=None):
        """
        The initialization function for HowIFeel
        :param is_pre_trained: If the model is pre-trained
        :param feature_extract: If True - Use transfer Learning
        :param output_dim: Number of classes for the dataset
        :param model_name: Which model to train, if None will be determined by Config file
        """
        super(HowDoIFeel, self).__init__()
        if model_name is None:
            self.model_name = cfg.MODEL
        else:
            self.model_name = model_name
        self.model = None
        self.input_size = 0
        self.is_pre_trained = is_pre_trained
        weights = 'DEFAULT' if is_pre_trained else None
        if self.model_name == cfg.RES_NET_18:
            """ Resnet18 """
            self.model = resnet18(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=self.model.conv1.kernel_size, stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=self.model.conv1.bias)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, output_dim, bias=False)  # replace the last FC layer
            self.input_size = 224
            self.is_pre_trained = True
        elif self.model_name == cfg.VGG_13:
            self.model = vgg13(weights=weights)
            set_parameter_requires_grad(self.model, feature_extract)
            if not cfg.RGB:
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=self.model.features[0].kernel_size, stride=self.model.features[0].stride, padding=self.model.features[0].padding, bias=self.features[0].conv1.bias)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, output_dim, bias=False)
            self.dropout = nn.Dropout(p=cfg.DROPOUT)
            self.input_size = 224
            self.is_pre_trained = True
        elif self.model_name == cfg.VIT:
            model_checkpoint = 'google/vit-base-patch16-224-in21k'
            self.model = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
            self.dropout = nn.Dropout(0.1)
            set_parameter_requires_grad(self.model, feature_extract)
            config = ViTConfig()
            self.classifier = (
                nn.Linear(config.hidden_size, output_dim)
            )
            self.is_pre_trained = True
        elif self.model_name == cfg.FACE_NET:
            self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=cfg.NUM_CLASSES)
        elif self.model_name == cfg.PERSONAL_1:
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
        elif self.model_name == cfg.PERSONAL_2:
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
        elif self.model_name == cfg.PERSONAL_3:
            self.features = nn.Sequential(
                                            nn.Conv2d(cfg.NUM_INPUT_CHANNELS, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(2),
                                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.DROPOUT),
                                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(2),
                                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.DROPOUT),
                                            nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(2),
                                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.DROPOUT)
            )
            self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                            nn.Flatten(),
                                            nn.Linear(64, cfg.NUM_CLASSES))
        elif self.model_name == cfg.PERSONAL_VGG:
            self.features = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=2, stride=2)
                                          )
            self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Linear(512 * 3 * 3, 4096),
                                            nn.Dropout(cfg.DROPOUT),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, 4096),
                                            nn.Dropout(cfg.DROPOUT),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, cfg.NUM_CLASSES))
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Forward function for the model
        :param x: Input
        :return: Model ouput for x, f(x)
        """
        if self.is_pre_trained:
            if self.model_name == cfg.VIT:
                x = self.model(x)['last_hidden_state']
                output = self.dropout(x[:, 0, :])
                return self.classifier(output)
            else:
                return self.model(x)
        else:
            if self.model_name == cfg.PERSONAL_1:
                out = self.features(x)
                out = nn.functional.dropout(out, p=cfg.DROPOUT, training=True)
                out = self.classifier(out)
            elif self.model_name == cfg.PERSONAL_VGG:
                out = self.features(x)
                # print(out.shape)
                # out = out.view(-1, 512 * 2 * 2)
                # out = self.avg_pool(out)
                # out = torch.flatten(out, 1)
                out = self.classifier(out)
            else:
                out = self.features(x)
                # out = out.view(out.size(0), -1)
                # out = nn.functional.dropout(out, p=cfg.DROPOUT, training=True)
                out = self.classifier(out)
            return out


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 20)  # number of layers will be between 1 and 20
    layers = []
    in_channels = cfg.NUM_INPUT_CHANNELS
    out_channels = None
    kernel_size = None
    for i in range(n_layers):
        out_channels = trial.suggest_int("n_units_l{}".format(i), 4, 128)  # number of units will be between 4 and 128
        kernel_size = trial.suggest_int(f'kernel_size_l{i}', 1, 6)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
        layers.append(nn.BatchNorm2d(out_channels)),
        layers.append(nn.ELU(inplace=True))
        in_channels = out_channels
    p = trial.suggest_float("dropout_l{1}", 0.1, 0.5)  # dropout rate will be between 0.1 and 0.5
    layers.append(nn.Flatten())
    layers.append(nn.Dropout(p))
    layers.append(nn.Linear(out_channels * kernel_size * kernel_size, 64))
    p = trial.suggest_float("dropout_l{2}", 0.1, 0.5)  # dropout rate will be between 0.1 and 0.5
    layers.append(nn.Dropout(p))
    layers.append(nn.Linear(64, cfg.NUM_CLASSES))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


