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
from yolov6.infer import run as run_inference
from run_monodepth import run
import wandb
from day_calibs import Cam_Cal

wandb.init(project = "Bounding Boxes detection")
cam_cal = Cam_Cal()


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
time_frame = 20
day = 3
object_name = "chair"

cam_ext = Cam_Cal.get_cal(day)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# compute depth maps
run(
    f"t00{time_frame}.000",
    "output_monodepth",
    "weights/dpt_large-midas-2f21e586.pt",
    "dpt_large",
    True,
)


for kid in kid_list:

  # SMPL parameters

  outputs, human_center,human_corners, object_center = run_inference(weights="saved_ckpt/yolov6l6.pt", source=f"t00{time_frame}.000/k{kid}.color.jpg", img_size=1280)

  x_pers_pos = [human_corners[0],human_corners[2]]
  y_pers_pos = [human_corners[1],human_corners[3]]
  x_obj_pos =  [object_center[0][0]]
  y_obj_pos = [object_center[0][1]]

  #print(y_pers_pos)
  #print(y_obj_pos)

  #output = joblib.load('t0021.000/person/fit02/person_fit.pkl') 
  #print(output.keys())
  smpl = get_smplh([f't00{time_frame}.000/person/fit02/person_fit.pkl'],"male" , "cpu")
  verts, jtr, tposed, naked = smpl()
  jtr = torch.cat((jtr[:, :22], jtr[:, 25:26], jtr[:, 40:41]), dim=1) # (N, 24, 3)
  jtr = jtr.unsqueeze(1).unsqueeze(1).unsqueeze(1) # (N, 1, 1, 1, J, 3)
  verts = torch.matmul(verts[0] - torch.Tensor(cam_ext[kid]["translation"]).reshape(1,-1,3) , torch.Tensor(cam_ext[kid]["rotation"]).reshape(3,3) )

  #print("JTR : ", jtr)


  # Camera intrinsic parameters and Visualization

  behave_verts, faces_idx = load_ply(f"t00{time_frame}.000/person/fit02/person_fit.ply")
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

  h5_file = h5py.File(f"t00{time_frame}.000/{object_name}/fit01/{object_name}_fit_k{kid}_sdf.h5", 'r')
  norm_params = h5_file['norm_params'][:].astype(np.float32)
  bbox = h5_file['sdf_params'][:].astype(np.float32)
  norm_params = torch.Tensor(norm_params)
  #print(norm_params)
  bbox = torch.Tensor(bbox).view(2, 3)

  bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
  bbox_center = (bbox[0] + bbox[1]) / 2.0 * norm_params[3] + norm_params[:3]
  bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)
  print(bbox)
  gt_obj_x = bbox[0]
  gt_obj_y = bbox[1]
  gt_obj_z = bbox[2]
  obj_dim = bbox[3]
  print("GT Object z mean position --> ", gt_obj_z)

  img = Image.open(f"input/k{kid}.color.jpg")
  convert_tensor = transforms.ToTensor()
  img_tensor = (convert_tensor(img).float() * 255).int()

  mask_person = Image.open(f"t00{time_frame}.000/k{kid}.person_mask.jpg")
  mask_tensor_p = convert_tensor(mask_person) > 0.5
  #print(mask_tensor_p.shape)
  #print(torch.unique(mask_tensor_p))

  mask_object = Image.open(f"t00{time_frame}.000/k{kid}.obj_rend_mask.jpg")
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
  pred_obj_x = (torch.min(verts[0,:,0]) * (x_pers_pos[1] - x_obj_pos[0]) + torch.max(verts[0,:,0]) * (x_obj_pos[0] - x_pers_pos[0])) / (x_pers_pos[1] - x_pers_pos[0]) #linear interpolation formula
  pred_obj_y = (torch.min(verts[0,:,1]) * (y_pers_pos[1] - y_obj_pos[0]) + torch.max(verts[0,:,1]) * (y_obj_pos[0] - y_pers_pos[0])) / (y_pers_pos[1] - y_pers_pos[0]) #linear interpolation formula
  pred_obj_z = ((torch.mean(img_tensor[mask_tensor_p])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (torch.mean(img_tensor[mask_tensor_o]))
  print(torch.max(verts[0,:,0]),torch.min(verts[0,:,0]))
  print(torch.max(verts[0,:,1]),torch.min(verts[0,:,1]))
  print("Pred Object x mean position --> ", pred_obj_x)
  print("Pred Object y mean position --> ", pred_obj_y)
  print("Pred Object z mean position --> ", pred_obj_z)
  #img_tensor = focal / img_tensor
  #img_tensor[img_tensor >= 5 * focal] = np.inf
  #img_tensor = o3d.geometry.Image(img_tensor[0].numpy())
  gt_obj_x *= -1
  gt_obj_y *= -1
  print("X Percentage Error between GT and Pred:", (abs((abs(pred_obj_x-gt_obj_x))/obj_dim)) * 100.0)
  print("Y Percentage Error between GT and Pred:", (abs((abs(pred_obj_y-gt_obj_y))/obj_dim)) * 100.0)
  print("Z Percentage Error between GT and Pred:", (abs((abs(pred_obj_z-gt_obj_z))/obj_dim)) * 100.0)
  

  img = Image.fromarray((img_tensor[0].numpy() * 255).astype(np.uint8))

  #print(torch.unique(img_tensor))
  #print(len(torch.unique(img_tensor)))

  #img.show()

  img = Image.open(f"t00{time_frame}.000/k2.depth.png")

  #img.show()

  convert_tensor = transforms.ToTensor()

  img_tensor = convert_tensor(img)

  #print(torch.unique(img_tensor))
  #print(len(torch.unique(img_tensor)))