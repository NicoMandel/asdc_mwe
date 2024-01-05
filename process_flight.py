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
from file_utils import read_config, get_flight_dir
from log_utils import log_exists, read_log, append_to_log
from visualise_bbox import visualize_object_predictions

IMAGE_EXTENSIONS += [".arw"]

def parse_args():
    fdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(fdir, 'data', 'inference', 'arw')

    conff = os.path.join(fdir, 'config', 'sahi_config.yaml')
    parser = ArgumentParser(description="Script for running the full inference process on a Single FLIGHT. Inference images using SAHI")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder", default=datadir)
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure")
    parser.add_argument("-m", "--model", default=None, help="Path to model file. If None given, will take first .pt file from <config> directory")
    parser.add_argument("--labels", action="store_true", help="If flag is set, will also store a directory with label files in yolov5 format.")
    parser.add_argument("-c", "--config", default=conff, type=str, help="Which file to use for configs. Defaults to sahi_config.yaml in the <config> directory")
    args = parser.parse_args()
    return vars(args)

def run_single_flight(
        flight_path : str,
        model,
        out_bool : bool,
        label_bool : bool,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
        padding_px,
        bbox_thickness,
        export_format,
        hide_labels
        ) :
    """
        Function to run a single site - reused by the "process_site" script. For consistency across scripts
    """
    log_list = []
    if out_bool:
        target_dir = get_flight_dir(flight_path)
        os.makedirs(target_dir, exist_ok=True)       
        print("Writing visuals to {}".format(target_dir))
        if log_exists(target_dir):
            log_list = read_log(target_dir)

    if label_bool:
        target_labels = get_flight_dir(flight_path, "labels")
        os.makedirs(target_labels, exist_ok=True)
        print("Writing labels to {}".format(target_labels))
    

    # Get single image result prediction
    image_iterator = list_files(
        directory=flight_path,
        contains=IMAGE_EXTENSIONS,
        verbose=2,
    )
    assert len(image_iterator) > 0, "No images in Folder: {} are you sure this is correct?".format(flight_path)

    elapsed_time = time.time()
    for image_path in tqdm(image_iterator, f"Performing inference on {flight_path}", leave=True):
        
        imgf, _ = os.path.splitext(os.path.basename(image_path))
        if imgf in log_list:
            print("Image with id {} already exists in log. Skipping".format(imgf))
            continue

        if image_path.endswith("ARW"):
            image_as_pil = read_arw_as_pil(image_path)
        else:
            image_as_pil = read_image_as_pil(image_path)

        result = get_sliced_prediction(
            image_as_pil,
            model,
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
            save_image(image, target_dir, imgf + "_vis",  export_format)
            append_to_log(target_dir, imgf)
        else:
            matplotlib.use('TkAgg')
            print(matplotlib.get_backend())
            plt.imshow(image)
            plt.show()

        # if saving labels:
        if label_bool:
            convert_pred_to_txt(result, target_labels, imgf)

    # Folder Statistics
    elapsed_time = time.time() - elapsed_time
    print("Took {:.2f} seconds to run {} images, so {:.2f} s / image".format(
        elapsed_time, len(image_iterator), elapsed_time / len(image_iterator)
    ))

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
    assert os.path.exists(source_dir), "{} does not exist. Double check".format(source_dir)
    
    label_bool=args["labels"]
    run_single_flight(
        source_dir,
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
    print("Finished processing site: {}".format(source_dir))