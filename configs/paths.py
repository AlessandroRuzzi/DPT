import socket
import getpass

username = getpass.getuser()
hostname = socket.gethostname()

dataroot = None
logroot = None
# BEHAVE dataset paths
behave_objs_path = None
behave_seqs_path = None
behave_calibs_path = None
# SMPL paths
smpl_assets_root = None
smpl_model_root = None

code = None

if hostname == 'helium-X550VX':
    pass
elif hostname == 'marvella':
    smpl_assets_root = "lib_smpl/assets"
    smpl_model_root = "/home/zahra/models/smpl/mano_v1_2/models" #"/BS/xxie2020/static00/mysmpl/smplh"
    code = "/home/zahra/workshop/hoi/interaction-learning/AutoSDF-code"
elif hostname == 'parham-laptop':
    pass
else:
    dataroot = 'data'
    logroot = 'logs'
    # BEHAVE dataset paths
    behave_objs_path = '/content/drive/MyDrive/Drive/Datasets/FullBehave/objects'
    behave_seqs_path = '/content/drive/MyDrive/Drive/Datasets/FullBehave/sequences'
    behave_calibs_path = '/content/drive/MyDrive/Drive/Datasets/FullBehave/calibs'
    smpl_assets_root = "lib_smpl/assets"
    smpl_model_root= "saved_ckpt/mano_v1_2/models" 
    code = "/content/drive/MyDrive/Drive/Models/interaction-learning/AutoSDF-code" # path to your project main folder

