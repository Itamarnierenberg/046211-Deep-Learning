import os


GPU_STR = 'mps'     # Apple GPU
CPU_STR = 'cpu'
RGB = False
NUM_INPUT_CHANNELS = 1
assert (RGB and NUM_INPUT_CHANNELS == 3) or (not RGB and NUM_INPUT_CHANNELS == 1)

RES_NET_18 = 'ResNet18'
VGG_13 = 'VGG-13'
PERSONAL = 'Personal'
MODEL = VGG_13  # ResNet18, VGG-13, PERSONAL
# Label Encoding

ANGRY = 0
DISGUST = 1
FEAR = 2
HAPPY = 3
SAD = 5
SURPRISE = 6
NEUTRAL = 4

CODE_TO_STR = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] # ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABELS_MAP = {ANGRY: 'Angry', DISGUST: 'Disgust', FEAR: 'Fear', HAPPY: 'Happy', SAD: 'Sad', SURPRISE: 'Surprise', NEUTRAL: 'Neutral'}
NUM_CLASSES = len(CODE_TO_STR)

# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory
DATASET_FOLDER = f'dataset'
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")
RESULTS_DIRECTORY = os.path.join("model")
MODEL_FILE = 'model.pth'
PLOT_FILE = 'plot.png'

# initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# specify the batch size, total number of epochs and the learning rate
BATCH_SIZE = 16
NUM_OF_EPOCHS = 50
LR = 1e-1
MOMENTUM = 0.9

FEATURE_EXTRACT = True