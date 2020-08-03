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
    videoPlayer("../../test1.mp4",['rgb','h'])