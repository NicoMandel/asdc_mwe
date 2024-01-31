# Trying to get things to work 
1. on `asdc` server infrastructure, for development users
2. for deployment users.

## Main Structure
There is code provided for two types of users:
### development users
are users who want to run a server infrastructure for further development and usage of models for research purposes
1. shell scripts are provided for wrapping `yolov5` code to run multiple training and testing instances in:
   1. [`train_test_model.sh`](./train_test_models.sh) trains and tests multiple model combinations on multiple datasets
   2. [`test_model.sh`](./test_model.sh) only tests a single provided model on a range of specified datasets
2. Jupyter notebooks for running inference with the `SAHI` package and previously developed models:
   1. [`inference_for_yolov5`](./inference_for_yolov5.ipynb) is a notebook which is dynamically updated from [`inference_for_yolov5.py`](./inference_for_yolov5.py). This is the basic notebook for running inference with `SAHI` and a previously developed model, which will only output visualisations
   2. [`inference_single_image`](./inference_for_yolov5.ipynb) is a notebook which is dynamically updated from [`inference_single_image.py`](./inference_single_image.py) which, despite its name, inferences each image individually and is thus capable of outputting label files!

### deployment users
are users who want to run inference on their local structure and want to process folders of large images. The scripts have been developed according to the workflow documented in [Issue#3](https://github.com/NicoMandel/asdc_mwe/issues/3). The general structure is that a `site` is split into multiple `flights`
1. [`process_flight.py`](./process_flight.py) is a script to run a single flight.
   1. provides command line utilities, so that it need only be run from the command line via `python process_flight.py -i <input_folder> [OPTIONS]`
   2. it contains the function `run_single_flight`, which runs a single flight directory and is in shared use with the second script
   3. includes (rudimentary) logging to skip images that have previously been processed
2. [`process_site.py`](./process_site.py) is a script to run multiple flights from a single site. 
   1. provides command line utilities, so that it need only be run from the command line via `python process_site.py -i <input_folder> [OPTIONS]`
   2. Looks for subdirectories containining the word `flight` (without spell-checking...) and will run a `run_single_flight` from [`process_flight`](./process_flight.py) for each of them.
   3. Includes (rudimentary) logging to skip already processed flights.
3. A [`config file`](./config/sahi_config.yaml) provides configurations for visualisations, such as minimum IoU and bounding box thickness. 

### Shared code
Code which is of importance to both types of users is provided in files post-fixed with `_utils.py`, at the current stage:
* [`log_utils.py`](./log_utils.py) for utilities pertaining to writing logs
* [`model_utils.py`](./model_utils.py) for utilities concerning model loading, file format handling, processing and visualisation
* [`file_utils.py`](./file_utils.py) for utilities concerning file handling, such as loading folders and listing files

## Default folder structure
`.pt` files are kept in the `config` subfolder.
Default data directories should be stored as subdirectories in the `data` folder

## Installation order
1. install miniconda (see online instructions)
2. install conda environment through `conda env install -f environment.yml`
3. install pip into conda environment through `conda install pip`
4. install rest of requirements through pip `pip install -r pipreq.txt --default-timeout=100`