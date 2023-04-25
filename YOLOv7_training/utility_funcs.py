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
def show_img(path, pred_labelpath='', labels=False,  clr1="blue", clr2="red"):

    img = Image.open(path)

    w,h =img.size
    fig_w = int(np.interp(w, [500, 5500], [6, 22]))
    fig_h = int(np.interp(h, [500, 5500], [6, 22]))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(np.asarray(img))
    plt.xticks([])
    plt.yticks([])

    if (labels == True) or (pred_labelpath != ''):

        labelpaths = [path]

        if pred_labelpath != '':
            labelpaths += [pred_labelpath]

        clrs = [clr1, clr2]

        for i in range(len(labelpaths)):
            try:
                labelpaths[i] = labelpaths[i].replace("images", "labels")
                labelpaths[i] = labelpaths[i][:-4]+".txt"
            except:
                labelpaths[i] = labelpaths[i][:-4]+".txt"
        
        
        if len(labelpaths) == 2:
            print(labelpaths)
            linewidths = [4, 1]
            bbxinc = [3, 0]

        else:
            linewidths = [1]
            bbxinc = [1]


        for i, path in enumerate(labelpaths):
            try:
                with open(path, mode="r") as f:
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

                        xmin = (xc_rel*w) - (0.5*wbb) - bbxinc[i]
                        xmax = (xc_rel*w) + (0.5*wbb) + bbxinc[i]
                        ymin = (yc_rel*h) - (0.5*hbb) - bbxinc[i]
                        ymax = (yc_rel*h) + (0.5*hbb) + bbxinc[i]


                        plt.plot([xmin, xmax], [ymin, ymin], color=clrs[i], linewidth=linewidths[i])
                        plt.plot([xmin, xmin], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                        plt.plot([xmax, xmax], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                        plt.plot([xmin, xmax], [ymax, ymax], color=clrs[i], linewidth=linewidths[i])
                        if len(labelpaths)==1:
                            plt.annotate(str(class_id), (xmin, ymin), c=clrs[i])
            except:
                print(f"label-file: '{path}' not found")
                
    plt.show()





def show_img2(imgdir, pred_labeldir='', num_imgs=1, labels=False,  clr1="blue", clr2="red"):
    # fancier version of show_img which can show up to 6 images in a subplot
    # can show true and predicted boundingboxes
    # if a list of imagepaths is input instead of an imagedirectory, specific images can be shown and compared
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    

    if type(imgdir) == str:     
        imgfiles = random.sample(os.listdir(imgdir), num_imgs)
        imgpaths = [imgdir+file for file in imgfiles]

    else: # a list of specific image paths is input. Note: the full path is required, not only filenames
        imgpaths = imgdir
        num_imgs = len(imgpaths)
        

    if (labels == True) or (pred_labeldir != ''):
        truelabelpaths = imgpaths.copy()
        for i in range(len(imgpaths)):
            try:
                truelabelpaths[i] = truelabelpaths[i].replace("images", "labels")
                truelabelpaths[i] = truelabelpaths[i][:-4]+".txt"
            except:
                truelabelpaths[i] = truelabelpaths[i][:-4]+".txt"

        labelpaths = [truelabelpaths]
    else:
        labelpaths = []

    
    if pred_labeldir != '':
        filenames = [path[path.rindex("/"):-4] for path in imgpaths]
        predlabelpaths = [pred_labeldir+filename+".txt" for filename in filenames]
        labelpaths = labelpaths + [predlabelpaths]

    
    img = Image.open(imgpaths[0])
    w,h =img.size
    fig_w = int(np.interp(w, [500, 5500], [5, 22]))
    fig_h = int(np.interp(h, [500, 5500], [5, 22]))
    

    if num_imgs == 1:
        # plot single image
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(np.asarray(img))
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        ncols = {1:1, 2:2, 3:3, 4:3, 5:3, 6:3}[num_imgs]
        nrows= {1:1, 2:1, 3:1, 4:2, 5:2, 6:2}[num_imgs]
        fig, ax = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
        ax = ax.ravel()
        for i in range(num_imgs if num_imgs <=3 else 6):
            if i < num_imgs:
                ax[i].imshow(np.asarray(Image.open(imgpaths[i])))   
            ax[i].set_xticks([])
            ax[i].set_yticks([])



    clrs = [clr1, clr2] # clr1: true labels and bounding boxes, clr2: predicted

    if len(labelpaths) == 2:  # with both true and predcited bbxs, increase linewidth and size for true
        linewidths = [4, 1]
        bbxinc = [3, 0]

    else:
        linewidths = [1]
        bbxinc = [1]


    for i, paths in enumerate(labelpaths):
        for j, path in enumerate(paths):
            try:
                with open(path, mode="r") as f:
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

                        xmin = (xc_rel*w) - (0.5*wbb) - bbxinc[i]
                        xmax = (xc_rel*w) + (0.5*wbb) + bbxinc[i]
                        ymin = (yc_rel*h) - (0.5*hbb) - bbxinc[i]
                        ymax = (yc_rel*h) + (0.5*hbb) + bbxinc[i]


                        if num_imgs > 1:
                            ax[j].plot([xmin, xmax], [ymin, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax[j].plot([xmin, xmin], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax[j].plot([xmax, xmax], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax[j].plot([xmin, xmax], [ymax, ymax], color=clrs[i], linewidth=linewidths[i])
                            if len(labelpaths)==1:
                                ax[j].annotate(str(class_id), (xmin, ymin), (xmin, ymin-3), c=clrs[i])
                        else:
                            ax.plot([xmin, xmax], [ymin, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax.plot([xmin, xmin], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax.plot([xmax, xmax], [ymax, ymin], color=clrs[i], linewidth=linewidths[i])
                            ax.plot([xmin, xmax], [ymax, ymax], color=clrs[i], linewidth=linewidths[i])
                            if len(labelpaths)==1:
                                ax.annotate(str(class_id), (xmin, ymin), (xmin, ymin-3), c=clrs[i])

                    if (num_imgs > 1) and (len(labelpaths) != 1) and (i == 0):
                        ax[j].text(0, -0.04, str(path), size=12, ha="left", transform=ax[j].transAxes, color=clrs[i])  #path to true label

                    elif (num_imgs > 1) and (len(labelpaths) != 1) and (i == 1):
                        ax[j].text(0, -0.08, str(path), size=12, ha="left", transform=ax[j].transAxes, color=clrs[i])  #path to predicted label
            
            except:
                if (num_imgs > 1) and (len(labelpaths) != 1) and (i == 0):
                    ax[j].text(0, -0.04, "Background image", size=12, ha="left", transform=ax[j].transAxes)  #path to true label
                    
                elif (num_imgs > 1) and (len(labelpaths) != 1) and (i == 1):
                    ax[j].text(0, -0.08, "No labels predicted", size=12, ha="left", transform=ax[j].transAxes)  #path to predicted label
    
    fig.tight_layout()




    
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
