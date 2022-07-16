import os
import glob
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from draw import init_img, drawBox2D, drawBox3D
from utils import readLabels, readCalibration, computeBox3D, computeOrientation3D


def make_args():
    parser = argparse.ArgumentParser(description="KITTI 3D object visualization")
    parser.add_argument("--root-dir", default="D:/datasets/kitti/", 
                        type=str, help="root of dataset")
    parser.add_argument("--dataset", default="training", 
                        type=str, help="which dataset")
    parser.add_argument("--output-dir", default="./outputs", 
                        type=str, help="outputs directory")
    parser.add_argument("--cam", default=2, 
                        type=int, help="which camera(2 = left color camera)")
    parser.add_argument("--pred-dir", default="D:/datasets/kitti/training/label_2", 
                        type=str, help="directory to predictions")
    return parser


def parse_idx(s: str):
  # example: D:/datasets/kitti/training/pred_2\000000.txt
  name = s.split("\\")[-1]
  idx = name.split(".")[0]
  for i, j in enumerate(idx):
    if j != "0":
      return int(idx[i:])
  return 0


def main():
  # options
  args = make_args().parse_args()
  root_dir = args.root_dir
  dataset = args.dataset
  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)
  cam = args.cam

  # get sub-directories
  image_dir = os.path.join(root_dir, dataset+'/image_'+str(cam))
  # label: ground-truth; pred: predictions
  label_dir = args.pred_dir
  label_files = glob.glob(os.path.join(label_dir, "*"))
  calib_dir = os.path.join(root_dir, dataset+'/calib')
  # get number of images for this dataset
  nimgs = len(glob.glob(os.path.join(image_dir, "*")))

  # main loop
  for label_file in tqdm(label_files):
    img_idx = parse_idx(label_file)
    image_file = "%s/%06d.png"%(image_dir, img_idx)
    img = Image.open(image_file)
    objects = readLabels(label_dir, img_idx)
    # load projection matrix
    P = readCalibration(calib_dir, img_idx, cam)
    # initialize figure
    fig, ax2D, ax3D = init_img(img=img, dataset=dataset, img_idx=img_idx, nimgs=nimgs)
    
    for obj in objects:
      # plot 2D bounding box
      drawBox2D(ax=ax2D, obj=obj)
      # plot 3D bounding box
      corners, face_idx = computeBox3D(obj, P)
      orientation = computeOrientation3D(obj, P)
      drawBox3D(ax3D=ax3D, obj=obj, corners=corners,
                face_idx=face_idx, orientation=orientation)
    output_file = "%s/%06d.png"%(output_dir, img_idx)
    plt.savefig(output_file)
    plt.close(fig)

if __name__ == "__main__":
  main()

    