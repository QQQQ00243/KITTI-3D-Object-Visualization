import numpy as np
import pandas as pd


class kitti_object:
    def __init__(
        self,
        type,      
        truncation,
        occlusion, 
        alpha,
        x1, y1, x2, y2,
        h, w, l, t, ry,
    ):
        # label, truncation, occlision
        self.type = type             # 'Car', 'Pedestrian', ...truncated pixel ratio ([0..1])
        self.truncation = truncation # truncated pixel ratio ([0..1])
        self.occlusion = occlusion   # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
        self.alpha = alpha           # object observation angle ([-pi..pi])

        # 2D bounding box in 0-based coordinates
        self.x1 = x1 # left
        self.y1 = y1 # top
        self.x2 = x2 # right
        self.y2 = y2 # bottom

        # 3D bounding box infomation
        self.h = h # box height
        self.w = w # box width
        self.l = l # box length
        self.t = t # location [x, y, z]
        self.ry = ry # yaw angle


def readCalibration(calib_dir, img_idx, cam):
    # load 3x4 projection matrix
    calib_file = "%s/%06d.txt"%(calib_dir, img_idx)
    calib_data = pd.read_csv(calib_file, header=None, sep=" ")
    P = calib_data.iloc[cam, 1:].to_numpy()
    return P.reshape([3, 4])


def readLabels(label_dir, img_idx):
    # parse input file
    objects = []
    label_file = "%s/%06d.txt"%(label_dir, img_idx)
    try:
        label_data = pd.read_csv(label_file, header=None, sep=" ")
    except:
        return []

    # extract information
    for i in range(len(label_data)):
        row = label_data.iloc[i, :].to_numpy()
        kitti = kitti_object(
            type=row[0],
            truncation=row[1],
            occlusion=row[2],
            alpha=row[3],
            x1=row[4], y1=row[5], x2=row[6], y2=row[7],
            h=row[8], w=row[9], l=row[10],
            t=row[11:14],
            ry=row[14],
        )
        objects.append(kitti)
    return objects


def projectToImage(pts_3D: np.ndarray, P: np.ndarray):
    # PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    # plane using the given projection matrix P.
    #
    # Usage: pts_2D = projectToImage(pts_3D, P)
    #   input: pts_3D: 3xn matrix
    #          P:      3x4 projection matrix
    #   output: pts_2D: 2xn matrix

    # project in image
    pts_2D = np.dot(P, np.concatenate(
        [pts_3D, np.ones([1, pts_3D.shape[1]])]
        )
    )
    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2,:]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2,:]
    return np.array(pts_2D[0:2, :], dtype=float)


def computeBox3D(obj, P):
    # takes an object and a projection matrix (P) and projects the 3D
    # bounding box into the image plane.

    # index for 3D bounding box faces
    face_idx = np.array([[0, 1, 5, 4],  # front face
                         [1, 2, 6, 5],  # left face
                         [2, 3, 7, 6],  # back face
                         [3, 0, 4, 7]])  # right face

    # compute rotational matrix around yaw axis
    R = [[+np.cos(obj.ry), 0, +np.sin(obj.ry)],
         [              0, 1,               0],
         [-np.sin(obj.ry), 0, +np.cos(obj.ry)]]

    # 3D bounding box dimensions
    l, w, h = obj.l, obj.w, obj.h

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # rotate and translate 3D bounding box

    corners_3D = np.dot(R, np.stack([x_corners, y_corners, z_corners]))
    corners_3D[0, :] = corners_3D[0, :] + obj.t[0]
    corners_3D[1, :] = corners_3D[1, :] + obj.t[1]
    corners_3D[2, :] = corners_3D[2, :] + obj.t[2]

    # only draw 3D bounding box for objects in front of the camera
    if np.any(corners_3D[2, :] < 0.1): 
        corners_2D = []
        return corners_2D, face_idx

    # project the 3D bounding box into the image plane
    corners_2D = projectToImage(corners_3D, P)
    return corners_2D, face_idx


def computeOrientation3D(obj, P):
    # takes an object and a projection matrix (P) and projects the 3D
    # object orientation vector into the image plane.

    # compute rotational matrix around yaw axis
    R = [[ np.cos(obj.ry), 0, np.sin(obj.ry)],
         [              0, 1,              0],
         [-np.sin(obj.ry), 0, np.cos(obj.ry)]]

    # orientation in object coordinate system
    orientation_3D = [[0.0, obj.l],
                      [0.0, 0.0],
                      [0.0, 0.0]]
    # rotate and translate in camera coordinate system, project in image
    orientation_3D      = np.dot(R, orientation_3D)
    orientation_3D[0, :] = orientation_3D[0, :] + obj.t[0]
    orientation_3D[1, :] = orientation_3D[1 ,:] + obj.t[1]
    orientation_3D[2, :] = orientation_3D[2 ,:] + obj.t[2]

    # vector behind image plane
    if np.any(orientation_3D[2, :]<0.1):
        orientation_2D = []
        return orientation_2D

    # project orientation into the image plane
    orientation_2D = projectToImage(orientation_3D, P)
    return orientation_2D


def main():
    calib_dir = "./calib"
    label_dir = "./pred_2"
    img_idx = 0
    cam = 2
    readLabels(label_dir=label_dir, img_idx=img_idx)


if __name__ =="__main__":
    main()