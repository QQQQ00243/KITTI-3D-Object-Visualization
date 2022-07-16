import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from utils import kitti_object, readLabels,\
    readCalibration, computeBox3D, computeOrientation3D


def init_img(img, dataset, img_idx, nimgs):
    # create figure based on image size
    fig = plt.figure(figsize=(0.01*img.size[0], 2*0.01*img.size[1]))
    ax2D = fig.add_axes([0, 0.5, 1, 0.5])
    ax3D = fig.add_axes([0, 0, 1, 0.5])

    # show image
    ax2D.imshow(img) 
    ax2D.axis("off")
    ax3D.imshow(img) 
    ax3D.axis("off")
  
    # title
    ax2D.text(x=0.5, y=0.98, s="2D Bounding Boxes", color="g", ha="center", va="top",
              fontsize=14, weight="bold", backgroundcolor="k", transform=ax2D.transAxes)
    ax3D.text(x=0.5, y=0.98, s="3D Bounding Boxes", color="g", ha="center", va="top",
              fontsize=14, weight="bold", backgroundcolor="k", transform=ax3D.transAxes)

    # legend
    ax2D.text(x=0, y=5, s="Not occluded", color="g", ha="left", va="top",
              fontsize=14, weight="bold", backgroundcolor="k")
    ax2D.text(x=0, y=30, s="Partly occluded", color="y", ha="left", va="top",
              fontsize=14, weight="bold", backgroundcolor="k")
    ax2D.text(x=0, y=60, s="Fully occluded", color="r", ha="left", va="top",
              fontsize=14, weight="bold", backgroundcolor="k")
    ax2D.text(x=0, y=90, s="Unknown", color="w", ha="left", va="top",
              fontsize=14, weight="bold", backgroundcolor="k")
    ax2D.text(x=0, y=120, s="Don't care region", color="c", ha="left", va="top",
              fontsize=14, weight="bold", backgroundcolor="k")

    # frame number
    ax2D.text(x=1, y=0.98, s="%s set frame %d/%d"%(dataset, img_idx, nimgs),
              color="g", ha="right", va="top", fontsize=14,
              weight="bold", backgroundcolor="k", transform=ax2D.transAxes)
    return fig, ax2D, ax3D


def drawBox2D(ax: plt.Axes, obj: kitti_object):
    # set styles for occlusion and truncation
    occ_col = ('g','y','r','w')
    trun_style = ('-','--')

    # draw regular objects
    if obj.type != "DontCare":
        # show rectangular bounding boxes
        trc = int(obj.truncation>0.1)
        pos = {"xy": [obj.x1, obj.y1], "width": obj.x2-obj.x1+1, 
               "height": obj.y2-obj.y1+1}
        
        rect = patches.Rectangle(**pos, edgecolor=occ_col[obj.occlusion],
                                linewidth=4, linestyle=trun_style[trc], facecolor="none")
        ax.add_patch(rect)
        
        rect = patches.Rectangle(**pos, edgecolor="b", facecolor="none", linewidth=0.5)
        ax.add_patch(rect)

        # draw label
        label_text = "%s\n%1.1f rad"%(obj.type, obj.alpha)
        x = (obj.x1+obj.x2)/2
        y = obj.y1
        ax.text(x=x, y=max(y-5, 40), s=label_text, color=occ_col[obj.occlusion],
            backgroundcolor="k", ha="center", va="bottom", weight="bold", fontsize=8)

    # draw don't care regions
    else:
        # draw dotted rectangle
        pos = {"xy": [obj.x1, obj.y1], "width": obj.x2-obj.x1+1,
               "height": obj.y2-obj.y1+1}
        rect = patches.Rectangle(**pos, edgecolor="c",
                                linewidth=2, linestyle="-")
        ax.add_patch(rect)


def drawBox3D(ax3D, obj: kitti_object, corners, face_idx, orientation):
    # set styles for occlusion and truncation
    occ_col    = ('g','y','r','w')
    trun_style = ('-','--')
    trc        = int(obj.truncation>0.1)

    # draw projected 3D bounding boxes
    corners = np.array(corners)
    if corners.size != 0:
        for f in range(4):
            x = np.array(corners[0, face_idx[f, :]])  #+ corners[0, face_idx[f, 0]]])
            y = np.array(corners[1, face_idx[f, :]])  #+ corners[1, face_idx[f, 0]]])
            ax3D.plot(x, y,
                      color=occ_col[obj.occlusion], linewidth=4,
                      linestyle=trun_style[trc])
            ax3D.plot(x, y, color='b', linewidth=0.5)
            
    # draw orientation vector
    orientation = np.array(orientation)
    if orientation.size != 0:
        x = np.concatenate([orientation[0, :], orientation[0, :]]) + 1
        y = np.concatenate([orientation[1, :], orientation[1, :]]) + 1
        ax3D.plot(x, y, color='w', linewidth=4)
        ax3D.plot(x, y, color='k', linewidth=2)

    # draw label
    label_text = "%s\n%1.1f rad"%(obj.type, obj.alpha)
    x = (obj.x1+obj.x2)/2
    y = obj.y1
    ax3D.text(x=x, y=max(y-5, 40), s=label_text, color=occ_col[obj.occlusion],
        backgroundcolor="k", ha="center", va="bottom", weight="bold", fontsize=8)


def main():
    cam = 2
    dataset="training"
    label_dir = "./label_2"
    image_dir = "./image_2"
    calib_dir = "./calib"
    img_idx=0
    img = Image.open("./image_2/000000.png")
    objects = readLabels(label_dir, img_idx)

    # get number of images for the dataset
    nimgs = len(glob.glob(os.path.join(image_dir, "*")))

    # load projection matrix
    P = readCalibration(calib_dir, img_idx, cam)

    # main loop
    # initialize figure
    ax2D, ax3D = init_img(img=img, dataset=dataset, img_idx=img_idx, nimgs=nimgs)

    # plot 2D bounding box
    drawBox2D(ax=ax2D, obj=objects[0])

    # plot 3D bounding box
    corners, face_idx = computeBox3D(objects[0], P)
    orientation = computeOrientation3D(objects[0], P)
    drawBox3D(ax3D=ax3D, obj=objects[0], corners=corners,
              face_idx=face_idx, orientation=orientation)
    plt.savefig("./0.png")
    plt.show()


if __name__ == "__main__":
    main()
