import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# This function takes an IMAGE path and displays it. The resulting figure size is proportional to the image size
# If labels=True, the functions tries to find the corresponding label based on the same filename and folder structure
# expected by the yolov7-model:
#
#       img_path ="/somedir/anotherdir/images/filename.png"
#       lbl_path ="/somedir/anotherdir/labels/filename.txt"
#
def show_img(path, labels=False, clr="blue"):


    img = Image.open(path)

    w,h =img.size
    fig_w = int(np.interp(w, [500, 5500], [6, 22]))
    fig_h = int(np.interp(h, [500, 5500], [6, 22]))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(np.asarray(img))
    plt.xticks([])
    plt.yticks([])

    if labels==True:

        try:
            labelpath = path.replace("images", "labels")
            labelpath = labelpath[:-4]+".txt"
        except:
            labelpath = labelpath[:-4]+".txt"

        try:
            with open(labelpath, mode="r") as f:
                lines = f.readlines()
                for line in lines:
                    content = line.split()
                    class_id = int(content[0])
                    xc_rel = float(content[1])
                    yc_rel = float(content[2])
                    w_rel = float(content[3])
                    h_rel = float(content[4])

                    wbb = w_rel*w
                    hbb = h_rel*h

                    xmin = (xc_rel*w) - (0.5*wbb)
                    xmax = (xc_rel*w) + (0.5*wbb)
                    ymin = (yc_rel*h) - (0.5*hbb)
                    ymax = (yc_rel*h) + (0.5*hbb)


                    plt.plot([xmin, xmax], [ymin, ymin], color=clr, linewidth=1)
                    plt.plot([xmin, xmin], [ymax, ymin], color=clr, linewidth=1)
                    plt.plot([xmax, xmax], [ymax, ymin], color=clr, linewidth=1)
                    plt.plot([xmin, xmax], [ymax, ymax], color=clr, linewidth=1)
                    plt.annotate(str(class_id), (xmin, ymin))
        except:
            print("no label file found")
        
    plt.show()

    
    
def move_images(src_paths, train, valid, test):

  if not os.path.isdir("yolov7/train/labels"):
      os.makedirs("yolov7/train/labels"), os.makedirs("yolov7/train/images")
  if not os.path.isdir("yolov7/valid/labels"):
      os.makedirs("yolov7/valid/labels"), os.makedirs("yolov7/valid/images")
  if not os.path.isdir("yolov7/test/labels"):    
      os.makedirs("yolov7/test/labels"), os.makedirs("yolov7/test/images")

  for i, file_set in enumerate([train, valid, test]):

    if len(file_set) == 0:
        continue

    for file in file_set:
        img_src = src_paths[0] + file
        txt_src = src_paths[1] + file[:-4]+".txt"

        if i == 0:
            img_dst = "yolov7/train/images/"+file
            txt_dst = "yolov7/train/labels/"+file[:-4]+".txt"
        elif i == 1:
            img_dst = "yolov7/valid/images/"+file
            txt_dst = "yolov7/valid/labels/"+file[:-4]+".txt"
        else:
            img_dst = "yolov7/test/images/"+file
            txt_dst = "yolov7/test/labels/"+file[:-4]+".txt"


        os.rename(img_src, img_dst)
        os.rename(txt_src, txt_dst)

  shutil.rmtree(src_paths[0])
  shutil.rmtree(src_paths[1])




    
def create_stratified_samples(csv, train_frac, valid_frac, test_frac, random_state=42):

    df = pd.read_csv(csv)
    os.remove(csv)

    dfg = pd.concat([df["image"], pd.get_dummies(df["class_id"], "cid")], axis=1)
    dfg = dfg.groupby(["image"]).max()

    cid_combination = []
    for _, row in dfg.iterrows():
        cid_combination.append(str(row["cid_-1.0"])+"-"+str(row["cid_0.0"])+"-" + str(row["cid_1.0"]))
    dfg["cid_combination"] = cid_combination


    x = dfg.index.values
    y = dfg["cid_combination"]
    
    if test_frac != 0:
        train_files, test_files, y_train, _ = train_test_split(x, y, test_size=test_frac, stratify=y, random_state=42)
        train_files, valid_files, _, _ = train_test_split(train_files, y_train, test_size=valid_frac/(1-test_frac), stratify=y_train, random_state=random_state)

    else:
        train_files, valid_files, _, _ = train_test_split(x, y, test_size=valid_frac, stratify=y, random_state=random_state)
        test_files = []

    print(f"n train: {len(train_files)}, n valid: {len(valid_files)}, n test: {len(test_files)} (ntot: {len(x)})")

    return train_files, valid_files, test_files




# adopted from function in nhttps://github.com/WongKinYiu/yolov7/blob/main/utils/torch_utils.py
def plot_results_simple(results_file, start=0, stop=0, annotate_best=False):

    fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['box-loss', 'obj-loss', 'cls-loss', '', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    results = np.loadtxt(results_file, usecols=[2, 3, 4, 12, 13, 14, 8, 9, 10, 11], ndmin=2).T
    n = results.shape[1]  # number of rows
    x = range(start, min(stop, n) if stop else n)

    for i in range(8):

        if i in [0, 1, 2]:
            y = results[i, x]
            y[y == 0] = np.nan  # don't show zero loss values
            ax[i].plot(x, y, marker='.', linewidth=1, markersize=2, c='orange')
            y = results[i+3, x]
            ax[i].plot(x, y, marker='.', linewidth=1, markersize=2)
            ax[i].set_title(s[i])
            ax[i].legend(['train', 'val'])

            # adjust yaxis
            avg_val = np.nanmean(results[i, x] + results[i+3, x])
            max_val = np.nanmax(results[i, x] + results[i+3, x])
            if 4*avg_val < max_val:
                ymax = max(np.nanquantile(results[i, x],0.95), np.nanquantile(results[i+3, x],0.95))
                ax[i].set_ylim((0,ymax))

        elif i in [4, 5, 6, 7]:
            y = results[i+2, x]
            ax[i].plot(x, y, marker='.', linewidth=1, markersize=2)
            ax[i].set_title(s[i])
            ax[i].legend(['val'])

            if annotate_best == True:
                max_yval = max(y)
                x_indx = [i for i, val in enumerate(y) if val == max(y)][0]

                ax[i].annotate(str(round(max_yval,3)), (x[x_indx], max_yval), c='red')

        elif i == 3:
            continue # skip one subplot


    plt.show()
    return
