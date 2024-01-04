import os.path
import yaml


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

def list_subdirectories(path : str, contains : list = []) -> list:
    sdl = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and contains in x.lower()]
    return sdl

def _get_child_dir(site_path : str, child_name : str = "Detections") -> str:
    sp = os.path.abspath(site_path)
    outp = os.path.join(sp, child_name)
    return outp

def get_labels_dir(site_path : str) -> str:
    return _get_child_dir(site_path, "labels")

def get_detections_dir(site_path : str) -> str:
    return _get_child_dir(site_path)

def get_flight_dir(flight_inputdir : str, child_name : str = "Detections") -> str:
    sp = os.path.normpath(flight_inputdir)
    flight_name = os.path.basename(sp)
    site_name = os.path.dirname(sp)
    site_outname = _get_child_dir(site_name, child_name)
    flight_target = os.path.join(site_outname, flight_name)
    return flight_target
    