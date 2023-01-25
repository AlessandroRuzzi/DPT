
from lib_smpl.smpl_generator import SMPLHGenerator
import numpy as np
import pickle
import torch
from lib_smpl.const import (
SMPLH_POSE_PRAMS_NUM, SMPLH_HANDPOSE_START,
)
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch, SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT
from lib_smpl.th_hand_prior import mean_hand_pose


def load_smpl(file):
    """
    load SMPL pose and shape
    :param file:
    :return: FrankMocap predicted pose (72, ) and shape (10, ) parameters
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def get_smpl_init(smpl_params_paths):

    """
    load FrankMocap pose prediction and initialize a SMPL-H model
    :param image_paths: input test image paths
    :param trans: smpl translation init
    :return: SMPLH model 
    """
    # load mocap data
    poses, betas, trans = [], [], []
    
    for file in smpl_params_paths:
        data = load_smpl(file)
        betas.append(data['betas'])
        poses.append(data['pose'])
        trans.append(data['trans'])

    smpl = SMPLHGenerator.get_smplh(np.stack(poses, 0),
                            np.stack(betas, 0),
                            np.stack(trans, 0), 
                            gender,
                            device=device)
    return smpl

def get_smplh_(poses, betas, trans, gender, device='cuda:0'):
    "generate smplh from a complete set of parameters"
    batch_sz = len(poses)
    pose_param_num = poses.shape[1]
    if pose_param_num != SMPLH_POSE_PRAMS_NUM:
        assert pose_param_num == 72, 'using unknown source of smpl poses'
        pose_init = torch.zeros((batch_sz, 156))
        pose_init[:, :pose_param_num] = torch.tensor(poses, dtype=torch.float32)
        pose_init[:, SMPLH_HANDPOSE_START:] = torch.tensor(mean_hand_pose(SMPL_ASSETS_ROOT), dtype=torch.float)
    else:
        pose_init = torch.tensor(poses, dtype=torch.float32)
    betas = torch.tensor(betas, dtype=torch.float32)
    smplh = SMPLPyTorchWrapperBatch(SMPL_MODEL_ROOT, batch_sz, betas, pose_init, trans,
                                    gender=gender, num_betas=10, hands=True, device=device).to(device)
    return smplh

def get_smplh(smpl_params_paths, gender, device):
    poses, betas, trans = [], [], []
    
    for file in smpl_params_paths:
        data = load_smpl(file)
        betas.append(data['betas'])
        poses.append(data['pose'])
        trans.append(data['trans'])

    smpl = get_smplh_(np.stack(poses, 0),
                            np.stack(betas, 0),
                            np.stack(trans, 0), 
                            gender,
                            device=device)
    return smpl

