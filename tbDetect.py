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


def segModel():
    """Deep learning segmentation model"""
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model


def dlSegment(img, model):
    """dlSegment(img, model) => masked image"""
    """Deep Learning segmentation algorithm"""
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
    mask[pred == 1] = [255, 165,0]
    return mask


def tennisBallDetect(video_path='../tennis_test.mp4'):
    """Detect a tennis ball in a video"""
    # get the video file input
    if not os.path.isfile(video_path):
        print(f'{video_path} doesn\'t exists')
        return
    vid = cv2.VideoCapture(video_path)


    # get the model
    model = segModel()

    ret, frm = vid.read()

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    size = frm.shape[0:2]
    dir, file = os.path.split(video_path)
    file, ext = os.path.splitext(file)
    vid_m_path = os.path.join(dir, file + "_mask" + ext)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    vw = cv2.VideoWriter(vid_m_path, fourcc, fps, size)
    fig, ax = plt.subplots()
    while ret:
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        mask = dlSegment(frm, model)
        displayHorz([frm, mask])
        vw.write(mask)
        ret, frm = vid.read()

    vid.release()
    vw.release()



if __name__ == "__main__":
    tennisBallDetect()