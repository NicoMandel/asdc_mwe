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

# import required functions, classes
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil

from utils import find_model_files, read_arw_as_pil, convert_pred_to_txt

IMAGE_EXTENSIONS += [".arw"]

def parse_args():
    fdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(fdir, 'data', 'inference', 'png')
    outdir = os.path.join(datadir, "labels")
    parser = ArgumentParser(description="File for creating labels on a folder of inference images using SAHI")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder", default=datadir)
    parser.add_argument("-o", "--output", required=False, help="which output folder to put the labels to", default=outdir)
    parser.add_argument("-m", "--model", default=None, help="Path to model file. If None given, will take first file from <config> directory")
    parser.add_argument("-c", "--confidence", default=0.01, type=float, help="Which confidence value to use as threshold for displaying. Defaults to 0.01")
    args = parser.parse_args()
    return vars(args)

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

    model_type = "yolov5"
    model_path = yolov5_model_path
    model_device = "cuda:0" # or 'cuda:0'
    model_confidence_threshold = 0.4

    slice_height = 1280
    slice_width = 1280
    overlap_height_ratio = 0.3
    overlap_width_ratio = 0.3

    source_image_dir = args["input"]
    target_dir = args["output"]
    # fullp = os.path.abspath(target_dir)
    # tgt_proj = 
    # # "demo_data/"
    # project = "SAHI"
    # name = "test_ds"

    # Get the model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov5',
        model_path=model_path,
        confidence_threshold=args["confidence"],
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
            slice_height = 1280,
            slice_width = 1280,
            overlap_height_ratio = 0.5,
            overlap_width_ratio = 0.5
        )
        imgf = os.path.basename(image_path).split(".")[0]
        convert_pred_to_txt(result, target_dir, imgf)

    elapsed_time = time.time() - elapsed_time
    print("Took {:.2f} seconds to run {} images, so {:.2f} s / image".format(
        elapsed_time, len(image_iterator), elapsed_time / len(image_iterator)
    ))