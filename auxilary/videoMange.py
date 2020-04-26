#!/usr/bin/python
import numpy as np
import cv2
import os
import sys
import scipy.io
import matplotlib.pyplot as plt
import skimage


def parseChannel(im, tags = []):
    """Creates a list of images and corresponding display settings"""
    if not im.ndim == 3 or not tags:
        return [im]
    ret = []
    for tag in tags:
        ret.append(captureChannel(im, tag))
    return ret


def captureChannel(im, ch):
    """Captures a requested channel out of an image with setting"""
    if ch is 'rgb':
        return im

    if ch in ['r','g','b']:
        sl = np.zeros(im.shape, dtype=np.uint8)
        clr = {'r': 2, 'g': 1, 'b': 0}

        sl[:, :, clr[ch]] = np.array(im[:, :, clr[ch]], dtype=np.uint8)
        return sl
    if ch in ['h','s','d']:
        sl = np.full(im.shape, 255,  dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        clr = {'h': 0, 's': 1, 'v': 2}
        sl[:, :, clr[ch]] = np.array(im[:, :, clr[ch]], dtype=np.uint8)
        return cv2.cvtColor(sl, cv2.COLOR_HSV2BGR)


def displayHorz(lst, fig, hstack=True):
    if hstack:
        #fig, ax = plt.subplots()

        #ax.imshow(np.hstack(lst))
        #plt.show()
        cv2.imshow("win", np.hstack(lst))


def videoInformation(cap, ret=[]):
    wdt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    hgt = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    bgr = cap.get(cv2.CAP_PROP_CONVERT_RGB)
    print(f'Video resolution is : {wdt}x{hgt}')
    print(f'at {fps} FPS')

    dict = {'width': wdt, 'height': hgt, 'fps': fps , 'bgr':bgr}
    lst = []
    for d in ret:
        lst.append(dict[d])
    if not lst:
        return
    if len(lst) is 1:
        return lst[0]
    return lst


def videoPlayer(fname, taglist=[]):
    if not os.path.isfile(fname):
        print(f'{fname} doesn\'t exists')
        return
    cap = cv2.VideoCapture(fname)
    fps = videoInformation(cap, ret=['fps'])
    framePeriod = int(np.floor(1000 / fps))
    wait = framePeriod

    ret, frm = cap.read()
    fig , ax = plt.subplots()
    while ret:

        displayHorz(parseChannel(frm, taglist), fig)
        if cv2.waitKey(wait) & 0xFF == 32:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('a'):
                wait = 0
            if key == 32:
                wait = framePeriod
        ret, frm = cap.read()

    cap.release()

if __name__ == "__main__":
    videoPlayer("../../test1.mp4",['rgb','h'])