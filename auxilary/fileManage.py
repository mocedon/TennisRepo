import os
import sys
import glob
import shutil
import pandas as pd
# from xml.dom import minidom
import xml.etree.ElementTree as ET


def readYolo(path):
    """readYolo(path)"""
    """Gets the data from a frame file"""
    data = pd.read_csv(path, sep=" ", header=None)
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



# def yolo2voc(pin, pout):


def pickObjYolo(pin, pout=None, obj=0):
    path = pin
    if pout:
        shutil.copytree(pin, pout)
        path = pout
    for f in os.listdir(path):
        if ".txt" in f and "classes" not in f:
            print(f)
            p = os.path.join(path, f)
            data = readYolo(p)
            sub = data[data["obj"] == obj]
            sub.to_csv(p, sep=" ", header=False, index=False)


if __name__ == "__main__":
    print("hello")
    orig = r'C:\Users\shura\OneDrive\EE\Semester 7\Poject A\raw_data\capture_right_2019_04_14_14_28_51.avi_from_0_to_10'
    path = r'C:\Users\shura\OneDrive\EE\Semester 7\Poject A\raw_data\backup'
    pickObjYolo(orig, path, 1)

    filesNumberFix(path, loc=1, l=3, ofs=600)
