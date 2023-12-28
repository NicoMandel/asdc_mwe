# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
# ---

import os
from tqdm import tqdm
import time
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

# import required functions, classes
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil

from utils import find_model_files, read_arw_as_pil, read_config
from visualise_bbox import visualize_object_predictions

IMAGE_EXTENSIONS += [".arw"]

def parse_args():
    fdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(fdir, 'data', 'inference', 'arw')
    # outdir = os.path.join(datadir, "visuals")
    outdir = None
    conff = os.path.join(fdir, 'config', 'sahi_config.yaml')
    parser = ArgumentParser(description="File for creating labels on a folder of inference images using SAHI")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder", default=datadir)
    parser.add_argument("-o", "--output", required=False, help="which output folder to put the labels to", default=outdir)
    parser.add_argument("-m", "--model", default=None, help="Path to model file. If None given, will take first .pt file from <config> directory")
    parser.add_argument("-c", "--config", default=conff, type=str, help="Which file to use for configs. Defaults to sahi_config.yaml in the <config> directory")
    args = parser.parse_args()
    return vars(args)

# Converting with custom functions:
# https://haobin-tan.netlify.app/ai/computer-vision/object-detection/coco-json-to-yolo-txt/
def convert_bbox_coco2yolo(img_w, img_h, pred_info):
    x_tl, y_tl, w, h = pred_info[0]
    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_c = x_tl + w / 2.
    y_c = y_tl + h / 2.

    x = x_c * dw
    y = y_c * dh

    x = x_c * dw
    y = y_c * dh
    w = w*dw
    h = h*dh
    return x, y, w, h, pred_info[1], pred_info[2]

def convert_pred(pred):
    img_w = pred.image_width
    img_h = pred.image_height
    # cats = [bbox.to_coco_annotation()["category_id"] for bbox in pred.object_prediction_list]
    pred_infos = [(bbox.to_coco_annotation().bbox, bbox.score.value, bbox.category.id) for bbox in pred.object_prediction_list]
    # pred_infos = [(an.bbox, an.score, an.category_id) for an in cc_annot]
    yolo_bboxes = [convert_bbox_coco2yolo(img_w, img_h, pred_info) for pred_info in pred_infos]
    if not yolo_bboxes: yolo_bboxes = None
    return yolo_bboxes

def convert_pred_to_txt(pred, target_dir, img_name : str = "labels"):
    # print(pred)
    yolo_bboxes = convert_pred(pred)
    outf = os.path.join(target_dir, img_name + ".txt")
    with open(outf, "w") as out:
        for x, y, w, h, score, category_id in yolo_bboxes:
            out.write(f"{category_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {score:.6f}\n")

def convert_pred_to_np(pred : np.ndarray) -> np.ndarray:
    yolo_bboxes = convert_pred(pred)
    return np.asarray(yolo_bboxes)

if __name__=="__main__":
    args = parse_args()

    # getting the model
    if args["model"] == None:
        fdir = os.path.abspath(os.path.dirname(__file__))
        confdir = os.path.join(fdir, "config")
        mfs = find_model_files(confdir)
        yolov5_model_path = mfs[0]
    else:
        yolov5_model_path = args["model"]


    # model settings
    model_type = "yolov5"
    model_path = yolov5_model_path
    model_device = "cuda:0" # or 'cuda:0'

    model_settings = read_config(args["config"])

    slice_height = model_settings["slice_height"]
    slice_width = model_settings["slice_width"]
    overlap_height_ratio = model_settings["overlap_height_ratio"]
    overlap_width_ratio = model_settings["overlap_width_ratio"]
    confidence = model_settings["model_confidence_threshold"]


    # visualisation settings
    padding_px = model_settings["padding_px"]
    bbox_thickness = model_settings["bbox_thickness"]
    export_format = model_settings["export_format"]
    hide_labels = model_settings["hide_labels"]
    source_image_dir = args["input"]
    target_dir = args["output"]

    # Get the model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov5',
        model_path=model_path,
        confidence_threshold=confidence,
        device="cuda:0", # or 'cuda:0'
    )

    # Get single image result prediction
    image_iterator = list_files(
        directory=source_image_dir,
        contains=IMAGE_EXTENSIONS,
        verbose=2,
    )

    print("Running on device:{}".format(
        torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU - No GPU used"
    ))
    elapsed_time = time.time()
    for ind, image_path in enumerate(
            tqdm(image_iterator, f"Performing inference on {source_image_dir}")
        ):
        if image_path.endswith("ARW"):
            image_as_pil = read_arw_as_pil(image_path)
        else:
            image_as_pil = read_image_as_pil(image_path)
        # test_img = "data/OHW/Inference/test_ds/DSC00009.png"
        result = get_sliced_prediction(
            image_as_pil,
            detection_model,
            slice_height = slice_height,
            slice_width = slice_width,
            overlap_height_ratio = overlap_height_ratio,
            overlap_width_ratio = overlap_width_ratio
        )
        imgf = os.path.basename(image_path).split(".")[0]
        
        # converting to numpy format
        bbox_np = convert_pred_to_np(result)
        
        visualize_object_predictions(
            np.ascontiguousarray(image_as_pil),
            object_prediction_list=bbox_np,
            rect_th=bbox_thickness,
            text_size=None,
            text_th=None,
            # color: tuple = None,
            hide_labels=hide_labels,
            hide_conf=False,
            output_dir=target_dir,
            file_name=imgf + "_vis",
            export_format=export_format,
            padding_px= padding_px,
        )

    elapsed_time = time.time() - elapsed_time
    print("Took {:.2f} seconds to run {} images, so {:.2f} s / image".format(
        elapsed_time, len(image_iterator), elapsed_time / len(image_iterator)
    ))