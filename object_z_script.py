import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from util import behave_camera_utils as bcu
import open3d as o3d
from lib_smpl.smpl_utils import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
from functools import partial
from pytorch3d.io import load_ply
import json
import os

cam_ext = [{
  "rotation": [
    -0.10153006011608547,
    0.019336641206071023,
    -0.9946445300707627,
    -0.011282787658231578,
    0.9997243968707996,
    0.02058710771274921,
    0.9947684884411452,
    0.013312573311902049,
    -0.10128390689705759
  ],
  "translation": [
    2.4029417954677847,
    0.0006033381196006956,
    2.3380179860610997
  ]
},{
  "rotation": [
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0
  ],
  "translation": [
    0.0,
    0.0,
    0.0
  ]
},{
  "rotation": [
    -0.017350607889147076,
    -0.026920036050998547,
    0.9994870024492014,
    0.00019428987272378874,
    0.9996373713560952,
    0.026927458842665578,
    -0.9998494479957091,
    0.0006613979823255235,
    -0.017339085771358613
  ],
  "translation": [
    -2.3940588412022525,
    -0.1613792875051818,
    2.3802714351100476
  ]
},{
  "rotation": [
    -0.9981333653659891,
    -0.015666431144359776,
    0.05902836503211517,
    -0.013929478981873385,
    0.9994611058176012,
    0.029723182419673092,
    -0.05946221118037236,
    0.028845465727886644,
    -0.9978137023254812
  ],
  "translation": [
    -0.08189590808177183,
    -0.10562289543456643,
    4.5437411730360555
  ]
}]

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

def show(img):
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(img)
    plt.show()

def show_projection(ver, img):
    #print(ver)
    for i in range(ver.shape[0]):
        img = cv2.circle(img, (ver[i, 0].int().item(), ver[i, 1].int().item()), 2, (255, 0, 0), 1)
    show(img)

kid_list = [0,1,2,3]

for kid in kid_list:

  # SMPL parameters

  #output = joblib.load('t0021.000/person/fit02/person_fit.pkl') 
  #print(output.keys())
  smpl = get_smplh(['t0021.000/person/fit02/person_fit.pkl'],"male" , "cpu")
  verts, jtr, tposed, naked = smpl()
  verts = torch.matmul(verts[0] - torch.Tensor(cam_ext[kid]["translation"]).reshape(1,-1,3) , torch.Tensor(cam_ext[kid]["rotation"]).reshape(3,3) )


  # Camera intrinsic parameters and Visualization

  behave_verts, faces_idx = load_ply("t0021.000/person/fit02/person_fit.ply")
  im = cv2.imread(f"input/k{kid}.color.jpg")
  behave_verts = behave_verts.reshape(-1, 3)
  #intrinsics = [bcu.load_intrinsics("t0021.000/calibs/intrinsics", i) for i in range(4)]
  intrinsics_2 = [load_intrinsics(os.path.join("t0021.000/calibs", "intrinsics"), i) for i in range(4)]
  calibration_matrix = intrinsics_2[kid][0]
  dist_coefs = intrinsics_2[kid][1]
  #print(calibration_matrix.shape, dist_coefs.shape)
  projector = get_local_projector(calibration_matrix, dist_coefs)
  show_projection(torch.from_numpy(projector(verts[0].detach().cpu().numpy())), im[:,:,::-1].copy())

  #print(torch.max(verts[0,:,2]))
  #print(torch.min(verts[0,:,2]))
  gt_per_z = torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0
  print("GT Person z mean position --> ", gt_per_z)

  h5_file = h5py.File(f"t0021.000/backpack/fit01/backpack_fit_k{kid}_sdf.h5", 'r')
  norm_params = h5_file['norm_params'][:].astype(np.float32)
  bbox = h5_file['sdf_params'][:].astype(np.float32)
  norm_params = torch.Tensor(norm_params)
  #print(norm_params)
  bbox = torch.Tensor(bbox).view(2, 3)

  bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
  bbox_center = (bbox[0] + bbox[1]) / 2.0 * norm_params[3] + norm_params[:3]
  bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)
  gt_obj_z = bbox[2]
  print("GT Object z mean position --> ", gt_obj_z)

  img = Image.open(f"input/k{kid}.color.jpg")
  convert_tensor = transforms.ToTensor()
  img_tensor = (convert_tensor(img).float() * 255).int()

  mask_person = Image.open(f"t0021.000/k{kid}.person_mask.jpg")
  mask_tensor_p = convert_tensor(mask_person) > 0.5
  #print(mask_tensor_p.shape)
  #print(torch.unique(mask_tensor_p))

  mask_object = Image.open(f"t0021.000/k{kid}.obj_rend_mask.jpg")
  mask_tensor_o = convert_tensor(mask_object) > 0.5
  #print(mask_tensor_o.shape)
  #print(torch.unique(mask_tensor_o))

  img = Image.open(f"output_monodepth/k{kid}.color.png")

  #img.show()

  convert_tensor = transforms.ToTensor()

  img_tensor = convert_tensor(img).float()
  #print(torch.unique(img_tensor))
  #print(len(torch.unique(img_tensor)))
  """
  batch,heigth,width = img_tensor.shape
  img_tensor = img_tensor.view(batch,-1)
  img_tensor -= img_tensor.min(1, keepdim=True)[0]
  img_tensor /= img_tensor.max(1, keepdim=True)[0]
  #img_tensor[0,2000:3100000] = 1.0
  img_tensor = img_tensor.view(batch,heigth,width)
  """

  #print(torch.mean(img_tensor))
  #print(torch.mean(img_tensor[mask_tensor_p]))
  #print(torch.mean(img_tensor[mask_tensor_o]))

  #print(((1-torch.mean(img_tensor[mask_tensor_o])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (1-torch.mean(img_tensor[mask_tensor_p])))
  pred_obj_z = ((torch.mean(img_tensor[mask_tensor_p])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (torch.mean(img_tensor[mask_tensor_o]))
  #pred_obj_z = (0.1 * ((torch.mean(img_tensor[mask_tensor_p]) - torch.mean(img_tensor[mask_tensor_o]))) + gt_per_z * (torch.mean(img_tensor[mask_tensor_o]) - torch.max(img_tensor))) / (torch.mean(img_tensor[mask_tensor_p]) - torch.max(img_tensor)) #linear interpolation formula
  print("Pred Object z mean position --> ", pred_obj_z)
  #img_tensor = focal / img_tensor
  #img_tensor[img_tensor >= 5 * focal] = np.inf
  #img_tensor = o3d.geometry.Image(img_tensor[0].numpy())

  print("Percentage Error between GT and Pred:", ((abs(pred_obj_z-gt_obj_z))/gt_obj_z) * 100.0)

  img = Image.fromarray((img_tensor[0].numpy() * 255).astype(np.uint8))

  #print(torch.unique(img_tensor))
  #print(len(torch.unique(img_tensor)))

  #img.show()

  img = Image.open("t0021.000/k2.depth.png")

  #img.show()

  convert_tensor = transforms.ToTensor()

  img_tensor = convert_tensor(img)

  #print(torch.unique(img_tensor))
  #print(len(torch.unique(img_tensor)))