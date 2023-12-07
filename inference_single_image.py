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

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil
from argparse import ArgumentParser


def parse_args():
    fdir = os.path.abspath(os.path.dirname(__file__))
    datadir = os.path.join(fdir, '..', 'data', 'inference')
    outdir = os.path.join(datadir, "labels")
    parser = ArgumentParser(description="File for creating labels on a folder of inference images using SAHI")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder", default=datadir)
    parser.add_argument("-o", "--output", required=False, help="which output folder to put the labels to", default=outdir)
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

def convert_pred_to_txt(pred, target_dir, img_name : str = "labels"):
    # print(pred)
    img_w = pred.image_width
    img_h = pred.image_height
    # cats = [bbox.to_coco_annotation()["category_id"] for bbox in pred.object_prediction_list]
    pred_infos = [(bbox.to_coco_annotation().bbox, bbox.score.value, bbox.category.id) for bbox in pred.object_prediction_list]
    # pred_infos = [(an.bbox, an.score, an.category_id) for an in cc_annot]
    yolo_bboxes = [convert_bbox_coco2yolo(img_w, img_h, pred_info) for pred_info in pred_infos]
    if not yolo_bboxes: return
    outf = os.path.join(target_dir, img_name + ".txt")
    with open(outf, "w") as out:
        for x, y, w, h, score, category_id in yolo_bboxes:
            out.write(f"{category_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {score:.6f}\n")    

if __name__=="__main__":
    args = parse_args()
    yolov5_model_path = "yolov5/ohw/combined_m_2/weights/best.pt"
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
        confidence_threshold=0.4,
        device="cuda:0", # or 'cuda:0'
    )

    # Get single image result prediction
    image_iterator = list_files(
        directory=source_image_dir,
        contains=IMAGE_EXTENSIONS,
        verbose=2,
    )

    for ind, image_path in enumerate(
            tqdm(image_iterator, f"Performing inference on {source_image_dir}")
        ):
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

