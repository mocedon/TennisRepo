import os
import sys
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
# from xml.dom import minidom
import xml.etree.ElementTree as ET


def readYolo(path):
    """readYolo(path)"""
    """Gets the data from a frame file"""
    data = pd.read_csv(path, sep=" ", header=None, dtype=str)
    data.columns = ["obj", "x", "y", "w", "h"]
    return data


def filesNumberFix(path=r'./', loc=0, l=4, ofs=0):
    """filesNumberFix(path, loc, l, eon=, ofs)"""
    for f in os.listdir(path):
        print(f)
        fn, ext = os.path.splitext(f)
        if not f[loc: loc + l].isdigit():
            break
        n = f[loc: loc + l]
        n = str(int(n) + ofs)
        n = n.zfill(l)
        print(n)
        name = os.path.join(path, n + ext)
        os.rename(os.path.join(path, f), name)


def voc2yolo(pin, pout):
    tree = ET.parse(pin)
    root = tree.getroot()
    imageW = float(root.find('image').get('width'))
    imageH = float(root.find('image').get('height'))
    labels = {'ball' : 0, 'person' : 1}
    for frm in root.iter('image'):
        count = str(frm.get('id')).zfill(len(str(len(root.findall('image')))))
        box = frm.find('box')
        s = ""
        if box != None:
            s += str(labels[box.get('label')])
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            s += " " + str((xbr + xtl) / (2 * imageW))[:8]
            s += " " + str((ybr + ytl) / (2 * imageH))[:8]
            s += " " + str(abs(xbr - xtl) / imageW)[:8]
            s += " " + str(abs(ybr - ytl) / imageH)[:8]
            print(s)
        fn = os.path.join(pout, count + ".txt")
        with open(fn, "w") as f:
            f.write(s)



def yolo2csv(pin, pout, label=0):
    csv = open(pout, 'w')
    frms = []
    for frm in glob.glob(pin + "/*.txt"):
        s = ""
        with open(frm, 'r') as frame:
            print(os.path.basename(frm))
            data = frame.readlines()
            for line in data:
                s += line if label == int(line.split(" ")[0]) else ""
        s = '\n' if  not s else s
        print('line is:{}'.format(s.strip('\n')))
        frms.append(s)
    frms[-1] = frms[-1].strip('\n')
    csv.writelines(frms)
    csv.close()



def pickObjYolo(pin, pout=None, obj=0):
    path = pin
    if pout:
        shutil.copytree(pin, pout)
        path = pout
    for f in glob.glob(os.path.join(path, "*.txt")):
        if ".txt" in f and "classes" not in f and (os.path.getsize(f) > 0):
            print(f)
            p = os.path.join(path, f)
            data = readYolo(p)
            sub = data[data["obj"] == str(obj)]
            sub.to_csv(p, sep=" ", header=False, index=False)


def downScale(path, scale=2):
    if os.path.exists(path):
        save_dir = path + "_ds"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img in glob.glob(path + "\*.png"):
            print(os.path.basename(img))
            im = cv2.imread(img)
            H , W , _ = im.shape
            im = cv2.resize(im, (int(W / scale), int(H / scale)))
            im_path = os.path.join(save_dir , os.path.basename(img))
            cv2.imwrite(im_path, im)



if __name__ == "__main__":
    path = r'..\..\yolo-ball-dataset-test_ds\labels'
    dest = r'C:\Users\shura\OneDrive - Technion\EE\Poject A\tennis\yolo-ball-dataset-rampup\result.csv'
    p = r'..\..\fix\labels'

    for i in l:
        f = open(i, "r")
        lines = f.readlines()
        f.close()
        f = open(i, "w")
        for line in lines:
            if float(line.split()[3]) > 0.01:
                print(line.strip('\n'))
                line = ""
            f.write(line)
        f.close()
