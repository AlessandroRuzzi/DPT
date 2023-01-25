import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from util import behave_camera_utils as bcu
import open3d as o3d
from lib_smpl.smpl_utils import *

cam_ext = {
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
}

# Camera intrinsic parameters

intrinsics = [bcu.load_intrinsics("t0021.000/calibs/intrinsics", i) for i in range(4)]
print(intrinsics[0])
focal = intrinsics[0][0][0][0]

# SMPL parameters

#output = joblib.load('t0021.000/person/fit02/person_fit.pkl') 
#print(output.keys())
smpl = get_smplh(['t0021.000/person/fit02/person_fit.pkl'],"male" , "cpu")
verts, jtr, tposed, naked = smpl()
print(verts.shape)
verts = torch.matmul(verts[0], torch.Tensor(cam_ext["rotation"]).reshape(3,3) ) + torch.Tensor(cam_ext["translation"]).reshape(1,-1,3)

print(verts.shape)

#print(torch.max(verts[0,:,2]))
#print(torch.min(verts[0,:,2]))
print(torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)

h5_file = h5py.File("t0021.000/backpack/fit01/backpack_fit_k2_sdf.h5", 'r')
norm_params = h5_file['norm_params'][:].astype(np.float32)
bbox = h5_file['sdf_params'][:].astype(np.float32)
norm_params = torch.Tensor(norm_params)
print(norm_params)
bbox = torch.Tensor(bbox).view(2, 3)

bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
bbox_center = (bbox[0] + bbox[1]) / 2.0 * norm_params[3] + norm_params[:3]
bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)
print(bbox)

img = Image.open("input/k2.color.jpg")
convert_tensor = transforms.ToTensor()
img_tensor = (convert_tensor(img).float() * 255).int()

mask_person = Image.open("t0021.000/k2.person_mask.jpg")
mask_tensor_p = convert_tensor(mask_person) > 0.5
print(mask_tensor_p.shape)
print(torch.unique(mask_tensor_p))

mask_object = Image.open("t0021.000/k2.obj_rend_mask.jpg")
mask_tensor_o = convert_tensor(mask_object) > 0.5
print(mask_tensor_o.shape)
print(torch.unique(mask_tensor_o))

img = Image.open("output_monodepth/k2.color.png")

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
print(torch.mean(img_tensor))
print(torch.mean(img_tensor[mask_tensor_p]))
print(torch.mean(img_tensor[mask_tensor_o]))

print(((1-torch.mean(img_tensor[mask_tensor_o])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (1-torch.mean(img_tensor[mask_tensor_p])))
print(((torch.mean(img_tensor[mask_tensor_p])) * (torch.min(verts[0,:,2]) + (torch.max(verts[0,:,2]) - torch.min(verts[0,:,2])) / 2.0)) / (torch.mean(img_tensor[mask_tensor_o])))
#img_tensor = focal / img_tensor
#img_tensor[img_tensor >= 5 * focal] = np.inf
#img_tensor = o3d.geometry.Image(img_tensor[0].numpy())

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