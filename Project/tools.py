import matplotlib.pyplot as plt
import torch
import Config as cfg
from torch.optim import lr_scheduler
import cv2


def print_hyper_params():
    print(f'[INFO] Hyper Parameters:')
    print(f'[INFO] \tTrain Portion = {cfg.TRAIN_SIZE}')
    print(f'[INFO] \tValidation Portion = {cfg.VAL_SIZE}')
    print(f'[INFO] \tNumber Of Epochs = {cfg.NUM_OF_EPOCHS}')
    print(f'[INFO] \tBatch Size = {cfg.BATCH_SIZE}')
    print(f'[INFO] \tLearning Rate = {cfg.LR}')
    print(f'[INFO] \tOptimizer = {cfg.OPTIMIZER}')
    print(f'[INFO] \tUsing Scheduler = {cfg.ENABLE_SCHEDULER}')
    if cfg.ENABLE_SCHEDULER:
        print(f'[INFO] \t\tScheduler Paitence = {cfg.SCHED_PATIENCE}')
        print(f'[INFO] \t\tMinimum Learning Rate = {cfg.MIN_LR}')
        print(f'[INFO] \t\tReduce Factor = {cfg.REDUCE_FACTOR}')
    print(f'[INFO] \tUsing Early Stopping = {cfg.ENABLE_EARLY_STOPPING}')
    if cfg.ENABLE_EARLY_STOPPING:
        print(f'[INFO] \t\tEarly Stopping Paitence = {cfg.EARLY_STOPPING_PATIENCE}')
        print(f'[INFO] \t\tMinimum Improvement = {cfg.MINIMUM_IMPROVEMENT}')


def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range - approximately... image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


def show_examples(train_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(cfg.LABELS_MAP[label])
        plt.axis("off")
        if not cfg.RGB:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(convert_to_imshow_format(img))
    plt.savefig(f'{cfg.RESULTS_DIRECTORY}/{cfg.DATA_SAMPLE_FILE}')


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # check if the width and height is specified
    if width is None and height is None:
        return image

    # initialize the dimension of the image and grab the
    # width and height of the image
    dimension = None
    (h, w) = image.shape[:2]

    # calculate the ratio of the height and
    # construct the new dimension
    if height is not None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))

    # resize the image
    resized_image = cv2.resize(image, dimension, interpolation=inter)

    return resized_image


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.grid(False)
    plt.savefig(f'{cfg.RESULTS_DIRECTORY}/{cfg.CONFUSION_MAT_FILE}')


class LRScheduler:
    """
    Check if the validation loss does not decrease for a given number of epochs
    (patience), then decrease the learning rate by a given 'factor'
    """

    def __init__(self, optimizer, patience=cfg.SCHED_PATIENCE, min_lr=cfg.MIN_LR, factor=cfg.REDUCE_FACTOR):
        """
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        :returns:  new_lr = old_lr * factor
        """

        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min",
                                                           patience=self.patience,
                                                           factor=self.factor,
                                                           min_lr=self.min_lr,
                                                           verbose=True)

    def __call__(self, validation_loss):
        self.lr_scheduler.step(validation_loss)


class EarlyStopping:
    """
    Early stopping breaks the training procedure when the loss does not improve
    over a certain number of iterations
    """

    def __init__(self, patience=cfg.EARLY_STOPPING_PATIENCE, min_delta=cfg.MINIMUM_IMPROVEMENT):
        """
        :param patience: number of epochs to wait stopping the training procedure
        :param min_delta: the minimum difference between (previous and the new loss)
                           to consider the network is improving.
        """

        self.early_stop_enabled = False
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def __call__(self, validation_loss):

        # update the validation loss if the condition doesn't hold
        if self.best_loss is None:
            self.best_loss = validation_loss

        # check if the training procedure should be stopped
        elif (self.best_loss - validation_loss) < self.min_delta:
            self.counter += 1
            print(f"[INFO] Early stopping: {self.counter}/{self.patience}... \n\n")

            if self.counter >= self.patience:
                self.early_stop_enabled = True
                print(f"[INFO] Early stopping enabled")

        # reset the early stopping counter
        elif (self.best_loss - validation_loss) > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0

