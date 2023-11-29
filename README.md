# Trying to get things to work on ASDC

## Processes attempted to get working
1. Train a model
2. Inference with a model

## Things to set up before Starting processes

### Train a model
1. Get Packages
	1. either from `requirements.txt` (pip) or `environment.yml` (conda)
2. Get Files into adjacent folders
	1. Image files
	2. Label files 
3. Set up a `data/<datasetname>.yml` file for yolo to train to
4. Clone the yolo repository from 
5. Run yolo training using `python train.py --data <path_to_dataset_definition>.yaml --project ohw --name <training_name> --img 1280 --batch-size 8 --epochs 300` 


### Running inference
#### Running on a folder to get visuals
1. Get the image files uploaded
2. Make changes to the `inference_for_yolov5.py` script to make it usable as a notebook
	* change model path to use the appropriate model
	* change folder path to run on the appropriate dataset
3. Run the script `inference_for_yolov5.py`
4. Inspect visuals in output folder

#### Running on a folder to get label files
1. Get the image files uploaded
2. Make changes to the `inference_single_image` notebook
	* argument parser?
	* set output directory?
3. inspect label files
