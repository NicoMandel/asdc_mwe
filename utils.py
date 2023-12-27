import os.path
from pathlib import Path
import rawpy
import numpy as np
from PIL import Image

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

def getlw() -> int:
    """
        Function to get the line width
    """
    return 15