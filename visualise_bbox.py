import os.path
import time
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_utils import getcolor, check_boundaries, save_image, show_image
np.set_printoptions(precision=6, suppress=True, linewidth=10000, edgeitems=30)


ALLOWED_EXTENSIONS = set(['jpg', 'JPEG', 'JPG', 'PNG', 'png', 'jpeg'])

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
        Straight from Yolo. utils/general.py
    """
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) - padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) - padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    padding_px : int = 0,
) -> tuple:
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    # text_th = text_th or max(rect_th - 1, 1)
    text_th =  max(rect_th - 1, 1)
    # set text_size for category names
    # text_size = text_size or rect_th / 3
    text_size =  rect_th / 3

    # add bboxes to image if present
    w = image.shape[1]
    h = image.shape[0]
    someret = xywhn2xyxy(object_prediction_list[:,1:], w, h, padh=padding_px, padw=padding_px)

    # bbox = object_prediction.bbox.to_xyxy()
    # todo: now iterate through every line in someret
    # someret = someret[np.newaxis, :] if someret.shape[0] == someret.size else someret 
    # print("Test Debug line")
    # return
    for row in someret:
        category_name = "OHW"       # row[0]
        score = row[4]

        # set color
        if color is None:
            color = getcolor(score)
        # set bbox points
        p1, p2 = (int(row[0]), int(row[1])), (int(row[2]), int(row[3]))
        p1 = check_boundaries(p1, w, h)
        p2 = check_boundaries(p2, w, h)
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )

        if not hide_labels:
            # arange bounding box text location
            label = f"{category_name}"

            if not hide_conf:
                label += f" {score:.2f}"

            w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # add bounding box text
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )
    elapsed_time = time.time() - elapsed_time
    return image, elapsed_time

def flatten_lists(l):
    return list(chain.from_iterable(l))

if __name__=="__main__":
    # potential todo - wrap this in a dataloader?
    fdir = os.path.abspath(os.path.dirname(__file__))
    # root_folder = os.path.abspath(os.path.join(fdir, '..', '..', "test_vis", "test_conf","test"))
    # root_folder = os.path.abspath(os.path.join(fdir, "data", "vis"))
    root_folder = os.path.abspath(os.path.join(fdir, "data", "inference", "png"))
    # root_folder = os.path.join(fdir, '..', 'data')
    # root_folder = os.path.join(fdir, "test_vis")
    # load images from folder
    # img_fs = os.path.join(root_folder, "inference")
    # img_fs = os.path.join(root_folder, "images")
    img_fs = os.path.join(root_folder)
    # load labels from other folder
    # labels = os.path.join(img_fs, "labels")
    labels = os.path.join(root_folder, "labels")

    # export directory
    # export_dir = os.path.join(root_folder, "visuals")
    export_dir = None

    plPath = Path(img_fs)
    lPath = Path(labels)
    img_list = [[x for x in plPath.glob(".".join(["*",fext]))] for fext in ALLOWED_EXTENSIONS]
    img_list = flatten_lists(img_list)
    label_list = list([x for x in lPath.glob("*.txt")])

    # image font settings
    visual_bbox_thickness: int = 15
    visual_text_size: float = None
    visual_text_thickness: int = None
    visual_hide_labels: bool = True
    visual_hide_conf: bool = False
    visual_export_format: str = "png"
    padding_px = 70
    # color = (57, 255, 20)  # model predictions in neon-green

    for lf in tqdm(label_list, desc="Label: ", leave=False):
        ln = os.path.basename(lf).split(".")[0]
        imgf = list([f for f in img_list if ln in str(f)])[0]
        img = read_image_as_pil(str(imgf))

        with open(lf, 'r') as f:
            lines = f.readlines()

        # test
        csvfor = np.genfromtxt(lf, delimiter=" ", ndmin=2)

        image, el_time = visualize_object_predictions(
            np.ascontiguousarray(img),
            object_prediction_list=csvfor,
            rect_th=visual_bbox_thickness,
            text_size=visual_text_size,
            text_th=visual_text_thickness,
            # color=color,
            hide_labels=visual_hide_labels,
            hide_conf=visual_hide_conf,
            padding_px=padding_px
        )
        if export_dir is not None:
            save_image(image, export_dir, file_name=ln+"_vis", export_format=visual_export_format)
        else:
            show_image(image)