# python3.11 -i ./Videos/TheDeparted.mp4 --model ./model/model.pth --prototxt ./faceDetection/architecture.txt --caffemodel ./faceDetection/weights.caffemodel

from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
from HowDoIFeel import HowDoIFeel
import torch.nn.functional as nnf
import tools
import numpy as np
import argparse
import torch
import cv2
import Config as cfg
from ffpyplayer.player import MediaPlayer
import time

# initialize the argument parser and establish the arguments required
args_dict = {'video': cfg.VIDEO_PATH, 'model': cfg.TRAINED_MODEL, 'prototxt': cfg.FACE_DETECTION, 'caffemodel': cfg.CAFFE_MODEL, 'confidence': 0.5}

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args_dict['prototxt'], args_dict['caffemodel'])

# check if gpu is available or not
device = cfg.GPU_STR if torch.has_mps else "cpu"

# dictionary mapping for different outputs
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprise"}

# load the emotionNet weights
is_pre_trained = False if cfg.MODEL == cfg.PERSONAL_1 or cfg.MODEL == cfg.PERSONAL_2 or cfg.PERSONAL_3 or cfg.PERSONAL_VGG else True
model = HowDoIFeel(is_pre_trained, model_name=cfg.VIDEO_MODEL)
model_weights = torch.load(args_dict["model"])
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# initialize a list of preprocessing steps to apply on each image during runtime
data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=cfg.NUM_INPUT_CHANNELS),
    Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
    ToTensor()
])

# initialize the video stream
vs = cv2.VideoCapture(args_dict['video'])
start_time = time.time()

# iterate over frames from the video file stream
while True:

    # read the next frame from the input stream
    (grabbed, frame) = vs.read()

    # check there's any frame to be grabbed from the steam
    if not grabbed:
        break

    # clone the current frame, convert it from BGR into RGB
    frame = tools.resize_image(frame, width=1500, height=1500)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # initialize an empty canvas to output the probability distributions
    canvas = np.zeros((300, 300, 3), dtype="uint8")

    # get the frame dimension, resize it and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

    # infer the blog through the network to get the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    # iterate over the detections
    for i in range(0, detections.shape[2]):

        # grab the confidence associated with the model's prediction
        confidence = detections[0, 0, i, 2]

        # eliminate weak detections, ensuring the confidence is greater
        # than the minimum confidence pre-defined
        if confidence > args_dict['confidence']:
            # compute the (x,y) coordinates (int) of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            # grab the region of interest within the image (the face),
            # apply a data transform to fit the exact method our network was trained,
            # add a new dimension (C, H, W) => (N, C, H, W) and send it to the device
            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)

            # infer the face (roi) into our pretrained model and compute the
            # probability score and class for each face and grab the readable
            # emotion detection
            predictions = model(face)
            prob = nnf.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            top_p, top_class = top_p.item(), top_class.item()

            # grab the list of predictions along with their associated labels
            emotion_prob = [p.item() for p in prob[0]]
            emotion_value = emotion_dict.values()
            # draw the probability distribution on an empty canvas initialized
            for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                prob_text = f"{emotion}: {prob * 100:.2f}%"
                width = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
                              (0, 0, 255), -1)
                cv2.putText(canvas, prob_text, (5, (i * 50) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # draw the bounding box of the face along with the associated emotion
            # and probability
            face_emotion = emotion_dict[top_class]
            face_text = f"{face_emotion}: {top_p * 100:.2f}%"
            cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(output, face_text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.05, (0, 255, 0), 2)

    # display the output to our screen
    cv2.imshow("Face", output)
    cv2.imshow("Emotion probability distribution", canvas)

    # break the loop if the `q` key is pressed
    elapsed = (time.time() - start_time) * 1000  # msec
    play_time = int(vs.get(cv2.CAP_PROP_POS_MSEC))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# destroy all opened frame and clean up the video-steam
cv2.destroyAllWindows()
vs.release()
