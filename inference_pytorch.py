from model.mfad import FAD_HAM_Net
import torch
import datetime
import cv2
import numpy as np
import argparse
import os
from dataset import TestDataAugmentation
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

switch = ['fake', 'real']


def predict(image_path, net, composed_transforms, device, y_true, y_pred):
    """
    A helper function to make a prediction

    Parameters
    ----------
    image_path : str
        A path to the image (Image name format: *_fake_* or *_real_*)
    net : FAD_HAM_Net
        A PyTorch model
    composed_transforms : TestDataAugmentation
        An object to make image prepropessing
    device : torch.device
        A device that wil be used
    y_true : list
        A list with true labels
    y_pred : list
        A list with predicted labels
    """

    image_x = cv2.imread(image_path)
    image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
    ta = datetime.datetime.now()
    image_x = composed_transforms(image=image_x)['image']
    image_x = image_x[np.newaxis, :].to(device)

    pred = net(image_x)
    tb = datetime.datetime.now()

    truth = image_path.split('/')[1].split("_")[1]
    if pred.detach().cpu() > 0.5:
        result = 'real'
    else:
        result = "fake"

    y_true.append(switch.index(truth))
    y_pred.append(switch.index(result))
    print(f"{image_path}; result - {result}")
    print('Inference time:', (tb - ta).total_seconds() * 1000)

    return y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a model')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', required=True)
    parser.add_argument('--device', type=str, help='Either cpu or cuda', required=True)
    parser.add_argument('--input-dirs', type=str, nargs='+', help="A list that contains dirs with input pictures "
                                                                  "in format *_fake_* or *_real_*"
                                                                  "(examples in the folder 'cropped')", required=True)
    parser.add_argument('--show-confusion', dest='confusion', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    else:
        raise Exception('Either cpu or cuda')

    # initialize model
    net = FAD_HAM_Net(pretrain=True, variant='resnet50').to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()
    # Initialize preprocessing
    composed_transforms = TestDataAugmentation()
    # Process images from input directories
    y_true = []
    y_pred = []
    for path in args.input_dirs:
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            y_true, y_pred = predict(image_path, net, composed_transforms, device, y_true, y_pred)

    print(f"Roc auc - {roc_auc_score(y_true, y_pred)}")
    print(f"Accuracy - {accuracy_score(y_true, y_pred)}")
    if args.confusion:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()
