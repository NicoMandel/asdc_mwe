import os
from tqdm import tqdm

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS, read_image_as_pil
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="File for creating labels on a folder of inference images using SAHI")
    parser.add_argument("-i", "--input", required=True, type=str, help="Location of the input folder")
    parser.add_argument("-o", "--output", required=True, help="which output folder to put the labels to")
    args = parser.parse_args()
    return vars(args)

# Converting with custom functions:
# https://haobin-tan.netlify.app/ai/computer-vision/object-detection/coco-json-to-yolo-txt/
def convert_bbox_coco2yolo(img_w, img_h, bbox):
    x_tl, y_tl, w, h = bbox
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
    return x, y, w, h
    
def convert_pred_to_txt(pred, target_dir, img_name : str = "labels"):
    # print(pred)
    img_w = pred.image_width
    img_h = pred.image_height
    # cats = [bbox.to_coco_annotation()["category_id"] for bbox in pred.object_prediction_list]
    cc_annot = [bbox.to_coco_annotation() for bbox in pred.object_prediction_list]
    bboxes = [an.bbox for an in cc_annot]
    cats = [an.category_id for an in cc_annot]
    yolo_bboxes = [convert_bbox_coco2yolo(img_w, img_h, bbox) for bbox in bboxes]
    if not yolo_bboxes: return
    cat = 0
    outf = os.path.join(target_dir, img_name + ".txt")
    with open(outf, "w") as out:
        for x, y, w, h in yolo_bboxes:
            out.write(f"{cat} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")    

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

