import os

# A GPU string that is known by pytorch. MPS is for an M1 Apple MacBook pro gpu
GPU_STR = 'mps'
# The default CPU string in case there's no GPU
CPU_STR = 'cpu'
# If this flag is set to True then the data will be converted to a 3 channel input data
RGB = True
NUM_INPUT_CHANNELS = 3
assert (RGB and NUM_INPUT_CHANNELS == 3) or (not RGB and NUM_INPUT_CHANNELS == 1)
# The initial default image size of the dataset
DEFAULT_SIZE = 224
# The image height and width to which the images will be resized to
IMAGE_HEIGHT = DEFAULT_SIZE
IMAGE_WIDTH = DEFAULT_SIZE
assert IMAGE_HEIGHT == IMAGE_WIDTH
# These are the models which are supported by the HowIFeel class
RES_NET_18 = 'ResNet18'
VGG_13 = 'VGG-13'
PERSONAL_1 = 'Personal_1'
PERSONAL_2 = 'Personal_2'
PERSONAL_3 = 'Personal_3'
PERSONAL_VGG = 'Personal_VGG'
VIT = 'ViT'
FACE_NET = 'FaceNet'
RESNET_WITH_BATCH_NORMALIZATION = 'ResNet18_Batch_Normalization'

# This parameter will determine which model will be trained, possible models listed above
MODEL = FACE_NET
# This parameter will enable the use of optuna
USE_OPTUNA = False

# Label Encoding
ANGRY = 0
DISGUST = 1
FEAR = 2
HAPPY = 3
SAD = 4
SURPRISE = 5
NEUTRAL = 6

CODE_TO_STR = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
LABELS_MAP = {ANGRY: 'Angry', DISGUST: 'Disgust', FEAR: 'Fear', HAPPY: 'Happy', SAD: 'Sad', SURPRISE: 'Surprise', NEUTRAL: 'Neutral'}
NUM_CLASSES = len(CODE_TO_STR)

# The dataset directory
DATASET_FOLDER = f'dataset'
# The train set directory
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
# The test set directory
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")
# The results directory
RESULTS_DIRECTORY = os.path.join("model")
# The name of the file to save the pytorch model to
MODEL_FILE = 'model.pth'
# The name of the plot file
PLOT_FILE = 'plot.png'
# The name of the ConfusionMatrix file
CONFUSION_MAT_FILE = 'ConfusionMatrix.png'
# The name of the Data Sample file
DATA_SAMPLE_FILE = 'DataSample.png'

# Train size and validation size
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# Training Parameters:
BATCH_SIZE = 64
NUM_OF_EPOCHS = 50
LR = 0.01
MOMENTUM = 0.9
DROPOUT = 0.1


# Optimizer
# Possible Optimizers: Adam, SGD
OPTIMIZER = 'SGD'
MAX_LEARNING_RATE = 0.001
GRAD_CLIP = None
WEIGHT_DECAY = 0.0001


# Scheduler Parameters:
# Possible Schedulers: ReduceLROnPlateau, OneCycleLR
ENABLE_SCHEDULER = True
SCHED_NAME = 'ReduceLROnPlateau'
SCHED_PATIENCE = 5          # Num of epochs to wait before updating the learning rate
MIN_LR = 1e-6               # When reaching this learning rate stop reducing
REDUCE_FACTOR = 0.75         # The factor which the learning rate will be reduced by

# Early Stopping Parameters:
ENABLE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 10    # Num of epochs to wait before stopping the training procedure
MINIMUM_IMPROVEMENT = 0         # the minimum difference between (previous and the new loss) to consider the network is improving.

FEATURE_EXTRACT = False

# Ensemble parameters
ENABLE_ENSEMBLE = True
ENSEMBLE_MODELS = [
    [VGG_13, './VGG13Model/model.pth'],
    [RES_NET_18, './ResNet18Model/model.pth']
]
ENSEMBLE_MODEL_1 = VGG_13
ENSEMBLE_MODEL_2 = RES_NET_18

# Detect From Video
VIDEO_PATH = './Videos/TheDeparted.mp4'
CAFFE_MODEL = './faceDetection/weights.caffemodel'
FACE_DETECTION = './faceDetection/architecture.txt'
TRAINED_MODEL = './VGG13Model/model.pth'
VIDEO_MODEL = PERSONAL_VGG
