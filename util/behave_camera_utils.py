
from functools import partial
import json
import os
import cv2
import numpy as np

def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(os.path.join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations

def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back

def load_intrinsics(intrinsic_folder, kid):
    with open(os.path.join(intrinsic_folder, f"{kid}", "calibration.json"), "r") as json_file:
            color_calib = json.load(json_file)['color']
            
            image_size = (color_calib['width'], color_calib['height'])
            focal_dist = (color_calib['fx'], color_calib['fy'])
            center = (color_calib['cx'], color_calib['cy'])
            
            calibration_matrix = np.eye(3)
            calibration_matrix[0, 0], calibration_matrix[1, 1] = focal_dist
            calibration_matrix[:2, 2] = center
            
            dist_coeffs = np.array(color_calib['opencv'][4:])

            return calibration_matrix, dist_coeffs, image_size

def load_kinect_params(config_folder, intrinsic_folder, kids, rotate=False):
    """
    return list of R, list of T, list of (calibration_matrix, dist_coeffs, img_size)
    """
    res = [load_intrinsics(intrinsic_folder, i) for i in kids]
    R, T = load_kinect_poses_back(config_folder, kids, rotate)
    return R, T, res

def global2local(verts, r, t):
    return verts @ r.T + t

def project_points(points, R, t, calibration_matrix, dist_coefs):
        """
        given points in the color camera coordinate, project it into color image
        points: (N, 3)
        R: (3, 3)
        t: (3)
        calibration_matrix: (3, 3)
        dist_coefs:(8)
        return: (N, 2)
        """
        return cv2.projectPoints(points[..., np.newaxis],
                                    R, t, calibration_matrix, dist_coefs)[0].reshape(-1, 2)

def get_local_projector(calibration_matrix, dist_coefs):
    return partial(project_points, R=np.eye(3), t=np.zeros(3), calibration_matrix=calibration_matrix, dist_coefs=dist_coefs)