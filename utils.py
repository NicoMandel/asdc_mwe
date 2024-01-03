import os.path

from pathlib import Path
import rawpy
import numpy as np
from PIL import Image
import yaml
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sahi import AutoDetectionModel

def find_model_files(dir : str) -> list:
    p = Path(dir)
    return list(p.glob("*.pt"))

def load_arw(fpath : str) -> np.ndarray:
    """
        Loading an ARW image. Thanks to Rob
    """
    raw = rawpy.imread(fpath)
    return raw.postprocess(use_camera_wb=True, output_bps=8)

def read_arw_as_pil(fpath : str) -> Image.Image:
    """ 
        function to return an ARW image in PIL format for SAHI
    """
    np_arr = load_arw(fpath)
    return Image.fromarray(np_arr)

def getcolor(conf : np.ndarray) -> np.ndarray:
    """
        function to map a confidence value into an RGB value.
        yellow is (255, 255, 0)
        blue is (0, 0, 255)
        black is (0, 0, 0)
    """
    outc = np.zeros((3), dtype=int)
    t =   255. * np.asarray([0. , 0., conf])
    outc = t.astype(int)
    return outc.tolist()

def _check_boundaries(p : int, ub : int, lb : int = 0) -> tuple:
    """
        Function to check whether a point is out of bounds of the image.
    """
    return np.clip(p, lb, ub)
    
def check_boundaries(p : tuple, w : int, h : int) -> tuple:
    """
        Function to check whether bbox corners are inside the image
        applied after padding
    """
    p1_checked = _check_boundaries(p[0], w)
    p2_checked = _check_boundaries(p[1], h)
    return (p1_checked, p2_checked)

def uniquify_dir(path : str) -> str:
    """
        Function to turn a directory path into a unique path if it already exists
    """
    bd = os.path.dirname(path)
    bn = os.path.basename(path)
    counter = 1
    # addstr = "youalmostdeletedyourdatayoudummy"
    while os.path.exists(path):
        path = os.path.join(bd, bn + "(" + str(counter) + ")")
        counter +=1 
    
    return path

def read_config(path : str) -> dict:
    """
        Utility function to read yaml config file and return
    """
    with open(path, 'r') as f:
        rd = yaml.safe_load(f)
    return rd

def save_image(image : np.ndarray, output_dir : str, file_name : str, export_format : str = "png"):
    # export image with predictions
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # save inference result
    save_path = str(Path(output_dir) / (file_name + "." + export_format))
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def show_image(image : np.ndarray):
    print(matplotlib.get_backend())

    plt.imshow(image)
    plt.show()

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
    return pred_info[2], x, y, w, h, pred_info[1]

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

def list_subdirectories(path : str, contains : list = []) -> list:
    sdl = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and contains in x.lower()]
    return sdl

def get_detections_dir(site_path : str) -> str:
    sp = os.path.normpath(site_path)
    bn = os.path.basename(sp)
    pd = os.path.dirname(sp)
    outp = os.path.join(pd, bn + "_Detections")
    # outp = uniquify_dir(outp)
    return outp

def get_flight_target_dir(flight_inputdir : str) -> str:
    sp = os.path.normpath(flight_inputdir)
    flight_name = os.path.basename(sp)
    site_name = os.path.dirname(sp)
    site_outname = get_detections_dir(site_name)
    flight_target = os.path.join(site_outname, flight_name)
    return flight_target

def get_model(path : str, confidence : float, model_type : str = "yolov5" , model_device : str = "cuda:0") -> AutoDetectionModel:
    if path == None:
        fdir = os.path.abspath(os.path.dirname(__file__))
        confdir = os.path.join(fdir, "config")
        mfs = find_model_files(confdir)
        model_path = mfs[0]
    else:
        model_path = path

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence,
        device=model_device, # or 'cuda:0'
    )

    return detection_model
    