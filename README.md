# Trying to get things to work on ASDC

## Default folder structure
`.pt` files are kept in the `config` subfolder.
Default data directories should be stored as subdirectories in the `data` folder

## Installation order
1. install miniconda (see online instructions)
2. install conda environment through `conda env install -f environment.yml`
3. install pip into conda environment through `conda install pip`
4. install rest of requirements through pip `pip install -r pipreq.txt --default-timeout=100`

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
6. Otherwise, use [this tutorial](https://github.com/ultralytics/yolov5/blob/master/tutorial.ipynb) from the Yolo Creators themselves how to use in a notebook

### Running inference
#### Running on a folder to get visuals
1. Get the image files uploaded
2. Get the model file uploaded 
3. Make changes to the `inference_for_yolov5.py` script to make it usable as a notebook
	* change model path to use the appropriate model
	* change folder path to run on the appropriate dataset
4. Run the script `inference_for_yolov5.py`
5. Inspect visuals in output folder

#### Running on a folder to get label files
1. Get the image files uploaded
	* are already in a `.png` format. usually require conversion from sony's ARW (using OpenCV custom script or commandline imagemagick)
2. Get the model file uploaded
3. Make changes to the `inference_single_image` notebook
	* argument parser?
	* set output directory?
4. inspect label files
