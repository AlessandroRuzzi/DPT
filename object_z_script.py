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
import random

wandb.init(project = "Bounding Boxes detection")

object_name_dict = {'backpack' : 'backpack', 'basketball' : 'sports ball', 'boxlarge' : 'boxlarge', 'boxlong' : 'boxlong', 'boxmedium' : 'boxmedium','boxsmall' : 'boxsmall',
                     'boxtiny' :'boxtiny' , 'chairblack' : 'chair','chairwood' : 'chair', 'keyboard' : 'keyboard' , 'monitor' : 'monitor', 'plasticcontainer': 'plasticcontainer', 
                    'stool' : 'stool', 'suitcase': 'suitcase', 'tablesmall' : 'tablesmall', 'tablesquare' : 'tablesquare', 'toolbox' : 'toolbox', 
                    'trashbin': 'trashbin', 'yogaball': 'sports ball', 'yogamat' : 'yogamat'}

def calc_lenght(img_tensor, verts, mask_tensor_p, mask_tensor_o, x_pers_pos, x_obj_pos, y_pers_pos, y_obj_pos, pred_obj_z, gt_lenght):

    pred_obj_x_top = (torch.min(verts[0,:,0]) * (x_pers_pos[1] - x_obj_pos[2]) + torch.max(verts[0,:,0]) * (x_obj_pos[2] - x_pers_pos[0])) / (x_pers_pos[1] - x_pers_pos[0]) #linear interpolation formula
    pred_obj_x_bottom = (torch.min(verts[0,:,0]) * (x_pers_pos[1] - x_obj_pos[1]) + torch.max(verts[0,:,0]) * (x_obj_pos[1] - x_pers_pos[0])) / (x_pers_pos[1] - x_pers_pos[0]) #linear interpolation formula    
    pred_obj_y_top = (torch.min(verts[0,:,1]) * (y_pers_pos[1] - y_obj_pos[2]) + torch.max(verts[0,:,1]) * (y_obj_pos[2] - y_pers_pos[0])) / (y_pers_pos[1] - y_pers_pos[0]) #linear interpolation formula
    pred_obj_y_bottom = (torch.min(verts[0,:,1]) * (y_pers_pos[1] - y_obj_pos[1]) + torch.max(verts[0,:,1]) * (y_obj_pos[1] - y_pers_pos[0])) / (y_pers_pos[1] - y_pers_pos[0]) #linear interpolation formula

    pred_obj_z_surface = ((torch.mean(img_tensor[mask_tensor_p])) * (torch.min(verts[0,:,2]))) / (torch.mean(img_tensor[mask_tensor_o]))


    lenght = max(abs(pred_obj_x_top-pred_obj_x_bottom),abs(pred_obj_y_top-pred_obj_y_bottom), (abs(pred_obj_z - pred_obj_z_surface) * 2).cuda())

    z = pred_obj_z_surface + lenght/2


    return z, lenght


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

def run_preprocessing(dataset_path):

    #sub_folders = os.walk(dataset_path)
    sequences_path = os.path.join(dataset_path,"sequences")
    calibs_path = os.path.join(dataset_path,"calibs")
    sub_folders = os.listdir(sequences_path)
    sub_folders.sort()

    kid_list = [0,1,2,3]
    error_dict = {'x' : 0, 'y' : 0, 'z': 0, 'z_optional' : 0, 'l': 0 , 'num_imgs' : 0}
    
    for folder in sub_folders:

        curr_folder_path = os.path.join(sequences_path,folder)

        strtok = [str(x) for x in folder.split('_') if x.strip()]
        day = int(strtok[0][-2:])
        object_name = object_name_dict[strtok[2]]
        print(day,object_name)

        time_folders = os.listdir(curr_folder_path)
        time_folders.sort()

        for time in time_folders:
            if time != "info.json":
                curr_time_folder_path = os.path.join(curr_folder_path, time)
                
                

                
                torch.backends.cudnn.benchmark = True
                # compute depth maps
                run(
                    curr_time_folder_path,
                    "output_monodepth",
                    "weights/dpt_large-midas-2f21e586.pt",
                    "dpt_large",
                    True,
                )

                for kid in kid_list:

                    cam_ext = json.load(open(os.path.join(calibs_path, f"Date0{day}/config/{kid}/config.json")))

                    # SMPL parameters
                    
                    outputs, human_center,human_corners, object_center = run_inference(weights="saved_ckpt/yolov6l6.pt", source=os.path.join(curr_time_folder_path, f"k{kid}.color.jpg"), img_size=1280)

                    #images = wandb.Image(outputs[:, :, ::-1], caption="Image with predicted bounding boxes")
                    #wandb.log({"Image YOLOv6" : images})

                    x_pers_pos = [human_corners[0],human_corners[2]]
                    y_pers_pos = [human_corners[1],human_corners[3]]
                    x_obj_pos =  [object_center[0][0],object_center[2][0],object_center[2][2]]
                    y_obj_pos = [object_center[0][1],object_center[2][1],object_center[2][3]]

                    
                    smpl = get_smplh([os.path.join(curr_time_folder_path ,'person/fit02/person_fit.pkl')], "female" , "cpu")
                    verts, jtr, tposed, naked = smpl()
                    jtr = torch.cat((jtr[:, :22], jtr[:, 25:26], jtr[:, 40:41]), dim=1) # (N, 24, 3)
                    jtr = jtr.unsqueeze(1).unsqueeze(1).unsqueeze(1) # (N, 1, 1, 1, J, 3)
                    verts = torch.matmul(verts[0] - torch.Tensor(cam_ext["translation"]).reshape(1,-1,3) , torch.Tensor(cam_ext["rotation"]).reshape(3,3) )



                    # Camera intrinsic parameters and Visualization

                    #behave_verts, faces_idx = load_ply(f"t00{time_frame}.000/person/fit02/person_fit.ply")
                    #im = cv2.imread(f"input/k{kid}.color.jpg")
                    #behave_verts = behave_verts.reshape(-1, 3)
                    #intrinsics = [bcu.load_intrinsics("t0021.000/calibs/intrinsics", i) for i in range(4)]
                    #intrinsics_2 = [load_intrinsics(os.path.join("t0021.000/calibs", "intrinsics"), i) for i in range(4)]
                    #calibration_matrix = intrinsics_2[kid][0]
                    #dist_coefs = intrinsics_2[kid][1]
                    #projector = get_local_projector(calibration_matrix, dist_coefs)
                    #show_projection(torch.from_numpy(projector(verts[0].detach().cpu().numpy())), im[:,:,::-1].copy())

                    gt_per_z = torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0
                    #print("GT Person z mean position --> ", gt_per_z)

                    h5_file = h5py.File(os.path.join(curr_time_folder_path,f"{object_name}/fit01/{object_name}_fit_k{kid}_sdf.h5"), 'r')
                    norm_params = h5_file['norm_params'][:].astype(np.float32)
                    bbox = h5_file['sdf_params'][:].astype(np.float32)
                    norm_params = torch.Tensor(norm_params)
                    bbox = torch.Tensor(bbox).view(2, 3)

                    bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
                    bbox_center = (bbox[0] + bbox[1]) / 2.0 * norm_params[3] + norm_params[:3]
                    bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)
                    #print(bbox)
                    gt_obj_x = bbox[0]
                    gt_obj_y = bbox[1]
                    gt_obj_z = bbox[2]
                    obj_dim = bbox[3]
                    #print("GT Object z mean position --> ", gt_obj_z)

                    

                    img = Image.open(f"input/k{kid}.color.jpg")
                    convert_tensor = transforms.ToTensor()
                    img_tensor = (convert_tensor(img).float() * 255).int()

                    mask_person = Image.open(os.path.join(curr_time_folder_path,f"k{kid}.person_mask.jpg"))
                    mask_tensor_p = convert_tensor(mask_person) > 0.5


                    mask_object = Image.open(os.path.join(curr_time_folder_path,f"k{kid}.obj_rend_mask.jpg"))
                    mask_tensor_o = convert_tensor(mask_object) > 0.5


                    img = Image.open(f"output_monodepth/k{kid}.color.png")


                    convert_tensor = transforms.ToTensor()

                    img_tensor = convert_tensor(img).float()

                    #print(((1-torch.mean(img_tensor[mask_tensor_o])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (1-torch.mean(img_tensor[mask_tensor_p])))
                    pred_obj_x = (torch.min(verts[0,:,0]) * (x_pers_pos[1] - x_obj_pos[0]) + torch.max(verts[0,:,0]) * (x_obj_pos[0] - x_pers_pos[0])) / (x_pers_pos[1] - x_pers_pos[0]) #linear interpolation formula
                    pred_obj_y = (torch.min(verts[0,:,1]) * (y_pers_pos[1] - y_obj_pos[0]) + torch.max(verts[0,:,1]) * (y_obj_pos[0] - y_pers_pos[0])) / (y_pers_pos[1] - y_pers_pos[0]) #linear interpolation formula
                    pred_obj_z = ((torch.mean(img_tensor[mask_tensor_p])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (torch.mean(img_tensor[mask_tensor_o]))
                    pred_obj_z_ref, lenght = calc_lenght(img_tensor, verts, mask_tensor_p, mask_tensor_o, x_pers_pos, x_obj_pos, y_pers_pos, y_obj_pos, pred_obj_z, obj_dim )
                    #pred_obj_z = ((torch.mean(img_tensor[0,x_pers_pos[0].int(): x_pers_pos[1].int(),y_pers_pos[0].int(): y_pers_pos[1].int()])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (torch.mean(img_tensor[0,object_center[2][0].int():object_center[2][2].int(),object_center[2][1].int():object_center[2][3].int()]))
                    
                    error_dict['x'] += (abs((abs(pred_obj_x-gt_obj_x))/obj_dim)) * 100.0
                    error_dict['y'] += (abs((abs(pred_obj_y-gt_obj_y))/obj_dim)) * 100.0
                    error_dict['z'] += (abs((abs(pred_obj_z-gt_obj_z))/obj_dim)) * 100.0
                    error_dict['z_optional'] += (abs((abs(pred_obj_z_ref-gt_obj_z))/obj_dim)) * 100.0
                    error_dict['l'] += (abs((abs(lenght - obj_dim))/obj_dim)) * 100.0
                    error_dict['num_imgs'] += 1

                    """
                    print(torch.max(verts[0,:,0]),torch.min(verts[0,:,0]))
                    print(torch.max(verts[0,:,1]),torch.min(verts[0,:,1]))
                    print("Pred Object x mean position --> ", pred_obj_x)
                    print("Pred Object y mean position --> ", pred_obj_y)
                    print("Pred Object z mean position --> ", pred_obj_z)
                    print("Pred Object lenght --> ", lenght)

                    gt_obj_x *= -1
                    gt_obj_y *= -1
                    print("X Percentage Error between GT and Pred:", (abs((abs(pred_obj_x-gt_obj_x))/obj_dim)) * 100.0)
                    print("Y Percentage Error between GT and Pred:", (abs((abs(pred_obj_y-gt_obj_y))/obj_dim)) * 100.0)
                    print("Z Percentage Error between GT and Pred:", (abs((abs(pred_obj_z-gt_obj_z))/obj_dim)) * 100.0)
                    print("Lenght Percentage Error between GT and Pred:", (abs((abs(lenght - obj_dim))/obj_dim)) * 100.0)
                    """
                print("-------------------------------------")
                print("X Error: ", error_dict['x'] / error_dict['num_imgs'])
                print("Y Error: ", error_dict['y'] / error_dict['num_imgs'])
                print("Z Error: ", error_dict['z'] / error_dict['num_imgs'])
                print("Z_optional Error: ", error_dict['z_optional'] / error_dict['num_imgs'])
                print("Lenght Error: ", error_dict['l'] / error_dict['num_imgs'])
                print("-------------------------------------\n")
                

                #break
        #break

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    dataset_path = "/data/xiwang/behave"
    run_preprocessing(dataset_path)