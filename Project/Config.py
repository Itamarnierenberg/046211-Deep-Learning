import os


GPU_STR = 'mps'     # Apple GPU
CPU_STR = 'cpu'
RGB = False
NUM_INPUT_CHANNELS = 1
DEFAULT_SIZE = 48
assert (RGB and NUM_INPUT_CHANNELS == 3) or (not RGB and NUM_INPUT_CHANNELS == 1)
IMAGE_HEIGHT = DEFAULT_SIZE
IMAGE_WIDTH = DEFAULT_SIZE
assert IMAGE_HEIGHT == IMAGE_WIDTH

RES_NET_18 = 'ResNet18'
VGG_13 = 'VGG-13'
PERSONAL_1 = 'Personal_1'
PERSONAL_2 = 'Personal_2'
VIT = 'ViT'
RESNET_WITH_BATCH_NORMALIZATION = 'ResNet18_Batch_Normalization'
MODEL = PERSONAL_1  # ResNet18, VGG-13, PERSONAL
NETWORK_CONFIG = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
# Label Encoding

ANGRY = 0
FEAR = 1
HAPPY = 2
SAD = 3
SURPRISE = 4
NEUTRAL = 5

CODE_TO_STR = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] # ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_MAP = {ANGRY: 'Angry', FEAR: 'Fear', HAPPY: 'Happy', SAD: 'Sad', SURPRISE: 'Surprise', NEUTRAL: 'Neutral'}
NUM_CLASSES = len(CODE_TO_STR)

# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory
DATASET_FOLDER = f'dataset'
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")
RESULTS_DIRECTORY = os.path.join("model")
MODEL_FILE = 'model.pth'
PLOT_FILE = 'plot.png'
CONFUSION_MAT_FILE = 'ConfusionMatrix.png'
DATA_SAMPLE_FILE = 'DataSample.png'

# Train size and validation size
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# Training Parameters:
BATCH_SIZE = 64
NUM_OF_EPOCHS = 50
LR = 0.1
MOMENTUM = 0.9
DROPOUT = 0.5

# Optimizer
OPTIMIZER = 'Adam'


# Scheduler Parameters:
ENABLE_SCHEDULER = True
SCHED_PATIENCE = 5          # Num of epochs to wait before updating the learning rate
MIN_LR = 1e-6               # When reaching this learning rate stop reducing
REDUCE_FACTOR = 0.3         # The factor which the learning rate will be reduced by

# Early Stopping Parameters:
ENABLE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 10    # Num of epochs to wait before stopping the training procedure
MINIMUM_IMPROVEMENT = 0         # the minimum difference between (previous and the new loss) to consider the network is improving.

FEATURE_EXTRACT = True
