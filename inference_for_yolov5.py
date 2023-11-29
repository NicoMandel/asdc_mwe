import os

# import required functions, classes
from sahi.predict import predict

# yolov5_model_path = 'models/yolov5s6.pt'
yolov5_model_path = "yolov5/ohw/combined_m_2/weights/best.pt"
model_type = "yolov5"
model_path = yolov5_model_path
model_device = "cuda:0" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 1280
slice_width = 1280
overlap_height_ratio = 0.3
overlap_width_ratio = 0.3

datadir = "data/OHW/Inference"
source_image_dir = os.path.join(datadir, "test_ds") 
# "demo_data/"
project = "SAHI"
name = "test_ds"

"""- Perform sliced inference on given folder:"""
result = predict(
    model_type=model_type,
    model_path=model_path,

    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    project=project,
    name=name,
    return_dict=True,
    novisual=True,
)
