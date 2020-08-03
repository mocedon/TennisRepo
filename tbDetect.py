import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from auxilary.videoMange import displayHorz
from auxilary.videoMange import videoPlayer
from auxilary.videoMange import vid2lst
from auxilary.videoMange import lst2vid


def segModel():
    """Deep learning segmentation model"""
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model


def dlSegment(img, model):
    """dlSegment(img, model) => masked image"""
    """Deep Learning segmentation algorithm"""
    print("Segmenting video")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
    # perform pre-processing
    input_tensor = preprocess(img) # Img normalization
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model

    # send to device
    input_batch = input_batch.to(device)
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    pred = np.where(output.argmax(0) == 37, 1, 0).astype('uint8')
    mask = np.zeros_like(img)
    mask[pred == 1] = [255, 165, 0]
    return mask


def tennisBallDetect(video_path, mask_path):
    """Detect a tennis ball in a video"""
    # get the video file input
    frms, info = vid2lst(video_path, ds=2, info=['fps', 'width', 'height'])

    # get the model
    model = segModel()

    mask = []
    for frm in frms:
        m = dlSegment(frm, model)
        plt.imshow(np.hstack([frm, m]))
        mask.append(m)

    lst2vid(mask, info, mask_path)


if __name__ == "__main__":
    videoPlayer("../test_tennis.avi", ['rgb', 'h', 'v'])
    tennisBallDetect("../tennis_test.mp4", "../tennis_test_mask.mp4")
    videoPlayer("../tennis_test_mask.mp4")