import os
from tqdm import tqdm
from argparse import ArgumentParser
import torch

from model_utils import get_model
from file_utils import read_config, list_subdirectories, get_detections_dir
from log_utils import log_exists, append_to_log, read_log
from process_flight import run_single_flight

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
    label_bool=args["labels"]
    source_dir = os.path.abspath(args["input"])
    
    process_dirs = list_subdirectories(source_dir, contains="flight")
    print("Processing site directory {} with {} flights".format(source_dir, len(process_dirs)))
    
    # check on found input directories
    assert process_dirs, "No subdirectories to process found. Please double check. Original input dir was {}".format(source_dir)

    # flight logging
    log_list = []
    if out_bool:
        det_dir = get_detections_dir(source_dir)
        if log_exists(det_dir):
            log_list = read_log(det_dir)

    for subdir in tqdm(process_dirs, leave=True):
        source_subdir = os.path.join(source_dir, subdir)
        if subdir in log_list:
            print("Already finished directory. {} in site {} Skipping".format(subdir, source_dir))
            continue
        
        run_single_flight(
            source_subdir,
            model=detection_model,
            out_bool=out_bool,
            label_bool=label_bool,
            slice_height = slice_height,
            slice_width = slice_width,
            overlap_height_ratio = overlap_height_ratio,
            overlap_width_ratio = overlap_width_ratio,
            padding_px = padding_px,
            bbox_thickness = bbox_thickness,
            export_format = export_format,
            hide_labels = hide_labels
        )

        if out_bool:
            append_to_log(det_dir, subdir)
