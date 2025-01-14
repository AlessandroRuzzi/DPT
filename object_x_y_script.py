import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
import cv2
import wandb
from yolov6.infer import run as run_inference

# Read Image

wandb.init(project = "Bounding Boxes detection")


#cv2.imshow("Original Image",im)

# Predict Bounding Box

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


# Print and Visualize Predictions

for i in range(4):
    im = cv2.imread(f"input/k{i}.color.jpg")
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    images = wandb.Image(out.get_image()[:, :, ::-1], caption="Image with predicted bounding boxes")
    wandb.log({"Image Detectron2" : images})


for i in range(4):
    #im = cv2.imread(f"input/k{i}.color.jpg")
    outputs, human_center, human_corners,object_center = run_inference(weights="saved_ckpt/yolov6l6.pt", source=f"input/k{i}.color.jpg", img_size=1280)

    print("MIN OBJECT BBOX-->", object_center[0])
    print("LABEL MIN BBOX -->", object_center[1])

    images = wandb.Image(outputs[:, :, ::-1], caption="Image with predicted bounding boxes")
    wandb.log({"Image YOLOv6" : images})