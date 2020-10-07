#!/usr/bin/python
import numpy as np
import cv2
import os
import sys
import scipy.io
import glob
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


def vid2lst(fname, ds=1 , info=[]):
    """Get a video parsed into frames on a list"""
    print(f'Getting file {fname}')
    if not os.path.isfile(fname):
        print(f'{fname} doesn\'t exists')
        return []
    cap = cv2.VideoCapture(fname)
    ret, frm = cap.read()
    lst = []
    while ret:
        if ds > 1:
            h, w = frm.shape[:2]
            frm = cv2.resize(frm, (w // ds , h // ds))
        lst.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        ret, frm = cap.read()

    ret = lst
    if info:
        vidInfo = fps = videoInformation(cap, ret=info)
        ret = [lst, vidInfo]

    cap.release()
    print("Got {:4d} frames".format(len(lst)))

    return ret


def lst2vid(lst, info, path):
    print("Taking {:4d} frames".format(len(lst)))
    [fps, w, h] = info
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(path, fourcc, int(fps), (int(w), int(h)))
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(lst)):
        out.write(lst[i])
    out.release()
    print("Frames saved as", path)


def lst2jpg(lst, path):
    print("Taking {:4d} frames".format(len(lst)))
    for i, img in enumerate(lst):
        fname = str(i).zfill(len(str(len(lst))))
        cv2.imwrite(os.path.join(path, fname + ".jpg"), img)


def jpg2lst(path):
    print("Taking frames from {}".format(path))
    lst = glob.glob(os.path.join(path, "*.jpg"))
    if not lst:
        lst = glob.glob(os.path.join(path, "*.png"))
    print("    There are {} frames".format(len(lst)))
    return lst


def drawRct(img, bb, dct):
    label = dct[bb[0]]
    h, w = img.shape[:2]
    x1 = int(w * (bb[1] - (bb[3] / 2)))
    x2 = int(w * (bb[1] + (bb[3] / 2)))
    y1 = int(h * (bb[2] - (bb[4] / 2)))
    y2 = int(h * (bb[2] + (bb[4] / 2)))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 6)
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
    # print('labelSize>>',labelSize)
    _x1 = x1
    _y1 = y1  # +int(labelSize[0][1]/2)
    _x2 = _x1 + labelSize[0][0]
    _y2 = y1 - int(labelSize[0][1])
    cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    return img


def captureChannel(im, ch):
    """Captures a requested channel out of an image with setting"""
    if ch is 'rgb':
        return im

    if ch in ['r','g','b']:
        sl = np.zeros(im.shape, dtype=np.uint8)
        clr = {'r': 0, 'g': 1, 'b': 2}

        sl[:, :, clr[ch]] = np.array(im[:, :, clr[ch]], dtype=np.uint8)
        return sl
    if ch in ['h','s','v']:
        sl = np.full(im.shape, 255,  dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        clr = {'h': 0, 's': 1, 'v': 2}
        sl[:, :, clr[ch]] = np.array(im[:, :, clr[ch]], dtype=np.uint8)
        return cv2.cvtColor(sl, cv2.COLOR_HSV2RGB)


def displayHorz(lst, fig, hstack=True):
    if hstack:
        #fig, ax = plt.subplots()

        #ax.imshow(np.hstack(lst))
        #plt.show()
        cv2.imshow("win", cv2.cvtColor(np.hstack(lst), cv2.COLOR_RGB2BGR))


def videoInformation(cap, ret=[]):
    wdt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    hgt = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    bgr = cap.get(cv2.CAP_PROP_CONVERT_RGB)
    print("Video resolution is : {:4d}x{:4d}".format(int(wdt), int(hgt)))
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


def video_with_BB(path):
    frms = jpg2lst(os.path.join(path, "images"))
    lbls = glob.glob(os.path.join(path, "labels/*.txt"))
    with open(os.path.join(path, "ball.yaml") , 'r') as yaml:
        in_block = False
        block = ""
        for line in yaml.readlines():
            if "names:" in line:
                in_block = True
            if in_block:
                if "]" in line:
                    in_block = False
                block += line
        block = block.replace('\n', '')
        block = block.replace(' ', '')
        block = block.replace('names:[', '')
        block = block.replace(']', '')
        block = block.replace('\'', '')
        lbl_dict = {i: lbl for i, lbl in enumerate(block.split(','))}
    print(lbl_dict)
    frm_bb = []

    for i in range(len(frms)):
        frm_bb.append(cv2.imread(frms[i]))
        lbl = lbls[i]
        with open(lbl, 'r') as f:
            for l in f.readlines():
                bb = [float(i) for i in l.split()]
                bb[0] = int(l.split()[0])
                frm_bb[i] = drawRct(frm_bb[i], bb, lbl_dict)
    info = [15, frm_bb[0].shape[1], frm_bb[0].shape[0]]
    lst2vid(frm_bb, info, os.path.join(path, "data_bb.mp4"))




def videoPlayer(fname, taglist=[]):

    frms, fps = vid2lst(fname, ds=4, info=['fps'])
    framePeriod = int(np.floor(1000 / fps))
    wait = framePeriod

    fig , ax = plt.subplots()
    for frm in frms:
        displayHorz(parseChannel(frm, taglist), fig)
        if cv2.waitKey(wait) & 0xFF == 32:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('a'):
                wait = 0
            if key == 32:
                wait = framePeriod

    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = r'C:\Users\shura\OneDrive - Technion\EE\Poject A\yolo-ball-dataset-train_ds'
    video_with_BB(path)