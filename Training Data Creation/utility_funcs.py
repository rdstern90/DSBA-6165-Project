#
# This function takes an IMAGE path and displays it. The resulting figure size is proportional to the image size
# If labels=True, the functions tries to find the corresponding label based on the same filename and folder structure
# expected by the yolov7-model:
#
#       img_path ="/somedir/anotherdir/images/filename.png"
#       lbl_path ="/somedir/anotherdir/labels/filename.txt"
#

def show_img(path, labels=False, clr="blue"):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

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