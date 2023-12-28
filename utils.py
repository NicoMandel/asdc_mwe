import os.path
from pathlib import Path
import rawpy
import numpy as np
from PIL import Image
import yaml

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
    t =   255. * np.asarray([10. / 255. , 10. / 255., 1. - conf])
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

def uniquify(path : str) -> str:
    """
        Function to turn a path into a unique path if it already exists
    """
    filen, ext = os.path.splitext(path)
    counter = 1
    # addstr = "youalmostdeletedyourdatayoudummy"
    while os.path.exists(path):
        path = filen + "(" + str(counter) + ")" + ext
        counter +=1 
    
    return path

def read_config(path : str) -> dict:
    """
        Utility function to read yaml config file and return
    """
    with open(path, 'r') as f:
        rd = yaml.safe_load(f)
    return rd