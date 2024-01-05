import os
from tqdm import tqdm
import time
from argparse import ArgumentParser
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import required functions, classes
import torch
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil

from model_utils import read_arw_as_pil, save_image, convert_pred_to_np, get_model, convert_pred_to_txt
from file_utils import get_detections_dir, get_labels_dir, read_config, list_subdirectories
from log_utils import log_exists, read_log, append_to_log
from visualise_bbox import visualize_object_predictions

IMAGE_EXTENSIONS += [".arw"]

def parse_args():
    fdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(fdir, 'data', 'inference', 'arw')

    conff = os.path.join(fdir, 'config', 'sahi_config.yaml')
    parser = ArgumentParser(description="Script for running the full inference process on a SITE. Inferences images using SAHI.")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder", default=datadir)
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure")
    parser.add_argument("-m", "--model", default=None, help="Path to model file. If None given, will take first .pt file from <config> directory")
    parser.add_argument("--labels", action="store_true", help="If flag is set, will also store a directory with label files in yolov5 format.")
    parser.add_argument("-c", "--config", default=conff, type=str, help="Which file to use for configs. Defaults to sahi_config.yaml in the <config> directory")
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":
    args = parse_args()

    # getting the model
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

    # Get the model
    print("Running on device:{}".format(
        torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU - No GPU used"
    ))
    detection_model = get_model(args["model"], confidence)

    # Folders
    out_bool = args["output"]           # used to create output folders
    source_dir = os.path.abspath(args["input"])
    
    process_dirs = list_subdirectories(source_dir, contains="flight")
    print("Processing site directory {} with {} flights".format(source_dir, len(process_dirs)))
    
    # check on found input directories
    assert process_dirs, "No subdirectories to process found. Please double check. Original input dir was {}".format(source_dir)

    # folder setup - only if out bool is set
    if out_bool:
        target_dir = get_detections_dir(source_dir)
        os.makedirs(target_dir, exist_ok=True)
        print("Writing to: {}".format(target_dir))

    # Label folders
    label_bool = args["labels"]
    if label_bool:
        label_dir = get_labels_dir(source_dir)
        os.makedirs(label_dir, exist_ok=True)
        print("Writing labels to: {}".format(label_dir))

    for source_subdir in tqdm(process_dirs, leave=True):
        log_list = []
        if out_bool:
            target_subdir = os.path.join(target_dir, source_subdir)
            os.makedirs(target_subdir, exist_ok=True)       # TODO - error catching here - if force flag not set, do not overwrite
            print("Writing visuals to {}".format(target_subdir))
            if log_exists(target_subdir):
                log_list = read_log(target_subdir)

        if label_bool:
            label_subdir = os.path.join(label_dir, source_subdir)
            os.makedirs(label_subdir, exist_ok=True)
            print("Writing labels to {}".format(label_subdir))
        

        source_image_dir = os.path.join(source_dir, source_subdir)
        # Get single image result prediction
        image_iterator = list_files(
            directory=source_image_dir,
            contains=IMAGE_EXTENSIONS,
            verbose=2,
        )

        elapsed_time = time.time()
        for ind, image_path in enumerate(
                tqdm(image_iterator, f"Performing inference on {source_image_dir}", leave=True)
            ):
            imgf = os.path.basename(image_path).split(".")[0]
            if imgf in log_list:
                print("Image with id {} already exists in log. Skipping".format(imgf))
                continue

            if image_path.endswith("ARW"):
                image_as_pil = read_arw_as_pil(image_path)
            else:
                image_as_pil = read_image_as_pil(image_path)

            result = get_sliced_prediction(
                image_as_pil,
                detection_model,
                slice_height = slice_height,
                slice_width = slice_width,
                overlap_height_ratio = overlap_height_ratio,
                overlap_width_ratio = overlap_width_ratio
            )
            
            # converting to numpy format
            bbox_np = convert_pred_to_np(result)
            
            image, _ = visualize_object_predictions(
                np.ascontiguousarray(image_as_pil),
                object_prediction_list=bbox_np,
                rect_th=bbox_thickness,
                text_size=None,
                text_th=None,
                # color: tuple = None,
                hide_labels=hide_labels,
                hide_conf=False,
                padding_px= padding_px,
            )
            
            if out_bool:
                save_image(image, target_subdir, imgf + "_vis",  export_format)
                append_to_log(target_subdir, imgf)
            else:
                matplotlib.use('TkAgg')
                print(matplotlib.get_backend())
                plt.imshow(image)
                plt.show()

            # if saving labels:
            if label_bool:
                convert_pred_to_txt(result, label_subdir, imgf)

        # Folder Statistics
        elapsed_time = time.time() - elapsed_time
        print("Took {:.2f} seconds to run {} images, so {:.2f} s / image".format(
            elapsed_time, len(image_iterator), elapsed_time / len(image_iterator)
        ))